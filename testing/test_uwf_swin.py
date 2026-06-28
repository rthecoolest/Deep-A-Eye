import os
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_uwf_late_fusion import UWFLateFusionDataset
from metrics_dr_full import compute_metrics, print_metrics_block

# ================= PATHS =================
TEST_CSV = "data/MMRDR_UWF_preprocessed_rgb/test_fusion.csv"
IMG_DIR = "data/MMRDR_UWF_preprocessed_rgb/all_images"
CKPT_PATH = "checkpoints/swin_uwf_late_fusion_best.pt"

# ================= MODEL =================
MODEL_NAME = "swin_small_patch4_window7_224.ms_in1k"

# ================= SETTINGS =================
NUM_CLASSES = 5
LESION_DIM = 7
BATCH_SIZE = 16
USE_CLAHE = True

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_ds = UWFLateFusionDataset(TEST_CSV, IMG_DIR, test_tf, USE_CLAHE)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_ds))


class SwinLateFusionModel(nn.Module):
    def __init__(self, model_name, num_classes=5, lesion_dim=7):
        super().__init__()

        self.image_encoder = timm.create_model(
            model_name,
            pretrained=False,
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

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

checkpoint = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for x, lesion, y, quality_weight, quality_raw, image_name in test_loader:
        x = x.to(device)
        lesion = lesion.to(device, dtype=torch.float32)
        y = y.to(device)

        logits = model(x, lesion)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

test_metrics = compute_metrics(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print("\n" + "=" * 90)
print("FINAL TEST RESULTS - SWIN LATE FUSION")
print("=" * 90)
print_metrics_block("test", test_metrics)

print("\nBest checkpoint metadata:")
print(f"Model: {checkpoint.get('model_name', 'N/A')}")
print(f"Run name: {checkpoint.get('run_name', 'N/A')}")
print(f"Fusion type: {checkpoint.get('fusion_type', 'N/A')}")
print(f"Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
print(f"Best validation QWK: {checkpoint.get('best_qwk', 'N/A')}")
print(f"Saved from stage: {checkpoint.get('stage', 'N/A')}")
print(f"Checkpoint path: {CKPT_PATH}")