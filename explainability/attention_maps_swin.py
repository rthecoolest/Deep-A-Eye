import os
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

from dataset_uwf_late_fusion import UWFLateFusionDataset


# ================= PATHS =================
TEST_CSV = "data/MMRDR_UWF_preprocessed_rgb/test_fusion.csv"
IMG_DIR = "data/MMRDR_UWF_preprocessed_rgb/all_images"
CKPT_PATH = "checkpoints/swin_uwf_late_fusion_best.pt"

OUTPUT_DIR = "attention_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= SETTINGS =================
MODEL_NAME = "swin_small_patch4_window7_224.ms_in1k"

NUM_CLASSES = 5
LESION_DIM = 7
USE_CLAHE = True

TARGET_TRUE_LABEL = 3

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ================= TRANSFORMS =================
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ================= DATASET =================
test_ds = UWFLateFusionDataset(
    TEST_CSV,
    IMG_DIR,
    test_tf,
    USE_CLAHE
)

print("Test samples:", len(test_ds))


# ================= MODEL =================
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

checkpoint = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()


# ================= FIND ONE CLASS 4 SAMPLE =================
sample_index = None

for i in range(len(test_ds)):
    _, _, true_label, _, _, image_name = test_ds[i]

    if true_label == TARGET_TRUE_LABEL:
        sample_index = i
        break

if sample_index is None:
    raise ValueError("No matching sample found")

print("Selected sample index:", sample_index)

image_tensor, lesion_vector, true_label, _, _, image_name = test_ds[sample_index]

input_tensor = image_tensor.unsqueeze(0).to(device)
lesion_vector = lesion_vector.unsqueeze(0).to(device)


# ================= HOOK =================
attention_maps = []


def hook_fn(module, input, output):
    attention_maps.append(output.detach().cpu())


# آخر بلوك داخل آخر layer
hook_handle = model.image_encoder.layers[-1].blocks[-1].register_forward_hook(hook_fn)


# ================= FORWARD =================
with torch.no_grad():
    logits = model(input_tensor, lesion_vector)
    pred_class = torch.argmax(logits, dim=1).item()

print("\nSelected image:", image_name)
print("True label:", true_label)
print("Predicted label:", pred_class)

hook_handle.remove()


# ================= GET ATTENTION =================
attn = attention_maps[0]

# shape:
# [B, H, W, C]

attn = attn[0]

# average channels
attn_map = attn.mean(dim=-1).numpy()

# normalize
attn_map = (attn_map - attn_map.min()) / (
    attn_map.max() - attn_map.min() + 1e-8
)

# resize to image size
attn_map = cv2.resize(attn_map, (224, 224))

# original image
rgb_img = image_tensor.permute(1, 2, 0).numpy()
rgb_img = np.clip(rgb_img, 0, 1)

# heatmap
heatmap = cv2.applyColorMap(
    np.uint8(255 * attn_map),
    cv2.COLORMAP_JET
)

heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = heatmap.astype(np.float32) / 255.0

overlay = 0.6 * rgb_img + 0.4 * heatmap
overlay = np.clip(overlay, 0, 1)


# ================= SAVE =================
save_path = os.path.join(
    OUTPUT_DIR,
    f"attention_{image_name}_true_{true_label}_pred_{pred_class}.png"
)

plt.imsave(save_path, overlay)

print("\nAttention map saved at:")
print(save_path)