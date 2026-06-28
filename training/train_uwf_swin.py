import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from dataset_uwf_late_fusion import UWFLateFusionDataset
from dataset_uwf_quality_split import build_quality_split_csv
from losses_dr_research import QualityAwareHybridLoss
from metrics_dr_full import compute_metrics, print_metrics_block

# ================= PATHS =================
TRAIN_CSV = "data/MMRDR_UWF_preprocessed_rgb/train_fusion.csv"
VAL_CSV = "data/MMRDR_UWF_preprocessed_rgb/val_fusion.csv"
TEST_CSV = "data/MMRDR_UWF_preprocessed_rgb/test_fusion.csv"
IMG_DIR = "data/MMRDR_UWF_preprocessed_rgb/all_images"
QUALITY_HIGH_CSV = "train_high_quality_fusion.csv"
QUALITY_LOW_CSV = "train_low_quality_fusion.csv"

# ================= MODEL =================
MODEL_NAME = "swin_small_patch4_window7_224.ms_in1k"
SAVE_DIR = "checkpoints"
RUN_NAME = "swin_uwf_late_fusion"

BEST_CKPT_PATH = os.path.join(SAVE_DIR, f"{RUN_NAME}_best.pt")
HISTORY_PATH = os.path.join(SAVE_DIR, f"{RUN_NAME}_history.json")

# ================= SETTINGS =================
NUM_CLASSES = 5
LESION_DIM = 7
BATCH_SIZE = 16
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 15
BASE_LR = 2e-5
WEIGHT_DECAY = 0.05
SEED = 42
FOCAL_GAMMA = 2.0
#we change it from 0.7 to 0.9 | date of change: Monday - May 4
ORDINAL_LAMBDA = 0.9
HIGH_QUALITY_RATIO = 0.30
USE_CLAHE = True

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= QUALITY SPLIT =================
build_quality_split_csv(
    TRAIN_CSV,
    IMG_DIR,
    QUALITY_HIGH_CSV,
    QUALITY_LOW_CSV,
    HIGH_QUALITY_RATIO
)

high_df = pd.read_csv(QUALITY_HIGH_CSV)
full_df = pd.read_csv(TRAIN_CSV)

print(f"High-quality train samples: {len(high_df)}")
print(f"Full train samples: {len(full_df)}")

# ================= TRANSFORMS =================
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================= DATASETS =================
train_high_ds = UWFLateFusionDataset(QUALITY_HIGH_CSV, IMG_DIR, train_tf, USE_CLAHE)
train_full_ds = UWFLateFusionDataset(TRAIN_CSV, IMG_DIR, train_tf, USE_CLAHE)
val_ds = UWFLateFusionDataset(VAL_CSV, IMG_DIR, eval_tf, USE_CLAHE)
test_ds = UWFLateFusionDataset(TEST_CSV, IMG_DIR, eval_tf, USE_CLAHE)

# ================= SAMPLER =================
def build_loader(dataset, df):
    labels = df["grade"].tolist()
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


train_high_loader = build_loader(train_high_ds, high_df)
train_full_loader = build_loader(train_full_ds, full_df)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ================= LATE FUSION MODEL =================
class SwinLateFusionModel(nn.Module):
    def __init__(self, model_name, num_classes=5, lesion_dim=7):
        super().__init__()

        self.image_encoder = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )

        image_feature_dim = self.image_encoder.num_features

        self.lesion_encoder = nn.Sequential(
            nn.Linear(lesion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, lesion):
        image_features = self.image_encoder(image)
        lesion_features = self.lesion_encoder(lesion)

        fused_features = torch.cat(
            [image_features, lesion_features],
            dim=1
        )

        logits = self.classifier(fused_features)

        return logits


model = SwinLateFusionModel(
    MODEL_NAME,
    num_classes=NUM_CLASSES,
    lesion_dim=LESION_DIM
)

model.to(device)
#Give more importance to the weaker middle classes: Class 2 and Class 3
class_weights = torch.tensor(
    [1.0, 1.1, 1.5, 1.6, 1.1],
    dtype=torch.float32
).to(device)

criterion_stage1 = QualityAwareHybridLoss(
    class_weights=class_weights,
    gamma=FOCAL_GAMMA,
    ordinal_lambda=0.9
)

criterion_stage2 = QualityAwareHybridLoss(
    class_weights=class_weights,
    gamma=FOCAL_GAMMA,
    ordinal_lambda=0.9
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=STAGE1_EPOCHS + STAGE2_EPOCHS
)

best_qwk = -1.0
best_epoch = -1
best_stage = ""
history = []

# ================= HELPERS =================
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def run_epoch(model, loader, criterion, optimizer=None, split_name="train"):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    all_true = []
    all_pred = []
    losses = []

    for x, lesion, y, q, _, _ in tqdm(loader, desc=split_name):
        x = x.to(device)
        lesion = lesion.to(device, dtype=torch.float32)
        y = y.to(device)
        q = q.to(device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x, lesion)
            loss = criterion(logits, y, q)

            if is_train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)

        losses.append(loss.item())
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    avg_loss = float(np.mean(losses))
    metrics = compute_metrics(all_true, all_pred)

    return avg_loss, metrics


def evaluate_and_print(model, criterion, loader, split_name):
    loss, metrics = run_epoch(
        model,
        loader,
        criterion,
        optimizer=None,
        split_name=split_name
    )

    print(f"\n{split_name} Loss: {loss:.4f}")
    print_metrics_block(split_name, metrics)

    return loss, metrics


def save_best_checkpoint(
    stage_name,
    epoch_number,
    train_loss,
    train_metrics,
    val_loss,
    val_metrics
):
    checkpoint = {
        "model_name": MODEL_NAME,
        "run_name": RUN_NAME,
        "fusion_type": "late_fusion_concat_image_features_and_lesion_features",
        "stage": stage_name,
        "best_epoch": epoch_number,
        "best_qwk": val_metrics["qwk"],

        "train_loss": train_loss,
        "train_metrics": train_metrics,

        "val_loss": val_loss,
        "val_metrics": val_metrics,

        "train_acc": train_metrics.get("acc"),
        "train_precision": train_metrics.get("macro_precision"),
        "train_recall": train_metrics.get("macro_recall"),
        "train_sensitivity": train_metrics.get("macro_sensitivity"),
        "train_specificity": train_metrics.get("macro_specificity"),
        "train_f1": train_metrics.get("macro_f1"),
        "train_qwk": train_metrics.get("qwk"),

        "val_acc": val_metrics.get("acc"),
        "val_precision": val_metrics.get("macro_precision"),
        "val_recall": val_metrics.get("macro_recall"),
        "val_sensitivity": val_metrics.get("macro_sensitivity"),
        "val_specificity": val_metrics.get("macro_specificity"),
        "val_f1": val_metrics.get("macro_f1"),
        "val_qwk": val_metrics.get("qwk"),

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "settings": {
            "num_classes": NUM_CLASSES,
            "lesion_dim": LESION_DIM,
            "batch_size": BATCH_SIZE,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "focal_gamma": FOCAL_GAMMA,
            "ordinal_lambda": ORDINAL_LAMBDA,
            "high_quality_ratio": HIGH_QUALITY_RATIO,
            "use_clahe": USE_CLAHE,
            "seed": SEED
        }
    }

    torch.save(checkpoint, BEST_CKPT_PATH)

    print("\n===== NEW BEST CHECKPOINT SAVED =====")
    print(f"Path: {BEST_CKPT_PATH}")
    print(f"Best epoch: {epoch_number}")
    print(f"Stage: {stage_name}")
    print(f"Best QWK: {val_metrics['qwk']:.4f}")
    print(f"Val Accuracy: {val_metrics.get('acc'):.4f}")
    print(f"Val Macro F1-score: {val_metrics.get('macro_f1'):.4f}")


# ================= STAGE 1 =================
for epoch in range(STAGE1_EPOCHS):
    current_epoch = epoch + 1

    print("\n" + "=" * 90)
    print(f"STAGE 1 | Epoch {current_epoch}/{STAGE1_EPOCHS}")

    train_loss, train_metrics = run_epoch(
        model,
        train_high_loader,
        criterion_stage1,
        optimizer=optimizer,
        split_name="train_stage1"
    )

    val_loss, val_metrics = evaluate_and_print(
        model,
        criterion_stage1,
        val_loader,
        "validation"
    )

    history.append({
        "stage": "stage1",
        "epoch": current_epoch,
        "train_loss": train_loss,
        "train_metrics": train_metrics,
        "val_loss": val_loss,
        "val_metrics": val_metrics
    })

    if val_metrics["qwk"] > best_qwk:
        best_qwk = val_metrics["qwk"]
        best_epoch = current_epoch
        best_stage = "stage1"

        save_best_checkpoint(
            "stage1",
            current_epoch,
            train_loss,
            train_metrics,
            val_loss,
            val_metrics
        )

    scheduler.step()

# ================= STAGE 2 =================
for epoch in range(STAGE2_EPOCHS):
    current_epoch = STAGE1_EPOCHS + epoch + 1

    print("\n" + "=" * 90)
    print(f"STAGE 2 | Epoch {epoch + 1}/{STAGE2_EPOCHS}")

    train_loss, train_metrics = run_epoch(
        model,
        train_full_loader,
        criterion_stage2,
        optimizer=optimizer,
        split_name="train_stage2"
    )

    val_loss, val_metrics = evaluate_and_print(
        model,
        criterion_stage2,
        val_loader,
        "validation"
    )

    history.append({
        "stage": "stage2",
        "epoch": current_epoch,
        "train_loss": train_loss,
        "train_metrics": train_metrics,
        "val_loss": val_loss,
        "val_metrics": val_metrics
    })

    if val_metrics["qwk"] > best_qwk:
        best_qwk = val_metrics["qwk"]
        best_epoch = current_epoch
        best_stage = "stage2"

        save_best_checkpoint(
            "stage2",
            current_epoch,
            train_loss,
            train_metrics,
            val_loss,
            val_metrics
        )

    scheduler.step()

# ================= SAVE HISTORY =================
with open(HISTORY_PATH, "w") as f:
    json.dump(make_json_safe(history), f, indent=4)

print("\nTraining complete")
print(f"Best epoch was {best_epoch}")
print(f"Best stage was {best_stage}")
print(f"Best QWK was {best_qwk:.4f}")
print(f"Best checkpoint saved at: {BEST_CKPT_PATH}")
print(f"Training history saved at: {HISTORY_PATH}")