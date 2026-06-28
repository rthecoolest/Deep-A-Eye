import os
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset_uwf_late_fusion import UWFLateFusionDataset


# ================= PATHS =================
TEST_CSV = "data/MMRDR_UWF_preprocessed_rgb/test_fusion.csv"
IMG_DIR = "data/MMRDR_UWF_preprocessed_rgb/all_images"
CKPT_PATH = "checkpoints/swin_uwf_late_fusion_best.pt"
OUTPUT_DIR = "gradcam_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= MODEL SETTINGS =================
MODEL_NAME = "swin_small_patch4_window7_224.ms_in1k"
NUM_CLASSES = 5
LESION_DIM = 7
USE_CLAHE = True
TARGET_TRUE_LABEL = 1

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= TRANSFORM =================
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

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

checkpoint = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()


# ================= WRAPPER FOR GRAD-CAM =================
class GradCAMWrapper(nn.Module):
    def __init__(self, model, lesion_vector):
        super().__init__()
        self.model = model
        self.lesion_vector = lesion_vector

    def forward(self, image):
        batch_size = image.size(0)

        lesion = self.lesion_vector.repeat(batch_size, 1)
        lesion = lesion.to(image.device, dtype=torch.float32)

        return self.model(image, lesion)


# ================= RESHAPE TRANSFORM FOR SWIN =================
def reshape_transform(tensor):
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)

    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


# ================= GET ONE CLASS SAMPLE =================
sample_index = None

for i in range(len(test_ds)):
    _, _, true_label, _, _, image_name = test_ds[i]

    if true_label == TARGET_TRUE_LABEL:
        sample_index = i
        break

if sample_index is None:
    raise ValueError(f"No image found with true label {TARGET_TRUE_LABEL}")

print("Selected sample index:", sample_index)

image_tensor, lesion_vector, true_label, quality_weight, quality_raw, image_name = test_ds[sample_index]

input_tensor = image_tensor.unsqueeze(0).to(device)
lesion_vector = lesion_vector.unsqueeze(0).to(device, dtype=torch.float32)

print("\nSelected image:", image_name)
print("True label:", true_label)


# ================= PREDICTION =================
with torch.no_grad():
    logits = model(input_tensor, lesion_vector)
    pred_class = torch.argmax(logits, dim=1).item()

print("Predicted label:", pred_class)


# ================= GRAD-CAM =================
cam_model = GradCAMWrapper(model, lesion_vector)

target_layers = [
    model.image_encoder.norm
]

targets = [
    ClassifierOutputTarget(pred_class)
]

cam = GradCAM(
    model=cam_model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]


# ================= PREPARE ORIGINAL IMAGE =================
rgb_img = image_tensor.permute(1, 2, 0).cpu().numpy()
rgb_img = np.clip(rgb_img, 0, 1)

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)


# ================= SAVE OUTPUT =================
safe_name = os.path.splitext(image_name)[0]
output_path = os.path.join(
    OUTPUT_DIR,
    f"gradcam_norm_{safe_name}_true_{true_label}_pred_{pred_class}.png"
)

cv2.imwrite(output_path, visualization_bgr)

print("\nGrad-CAM saved at:")
print(output_path)