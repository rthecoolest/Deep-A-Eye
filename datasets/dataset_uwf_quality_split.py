import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def clean_image_name(image_name):
    image_name = str(image_name).strip().split("/")[-1]

    if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        image_name += ".png"

    return image_name


def compute_quality_score_from_gray(gray):
    gray = gray.astype(np.float32)

    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    contrast = gray.std()
    brightness = gray.mean()

    sharpness_norm = min(lap_var / 500.0, 1.0)
    contrast_norm = min(contrast / 64.0, 1.0)

    brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
    brightness_score = max(0.0, brightness_score)

    quality_raw = 0.4 * sharpness_norm + 0.4 * contrast_norm + 0.2 * brightness_score

    quality_weight = 0.8 + 0.4 * quality_raw

    return float(quality_weight), float(quality_raw)


class UWFQualityDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None, use_clahe=True):
        self.df = pd.read_csv(csv_path, sep=None, engine="python")
        self.images_dir = images_dir
        self.transform = transform
        self.use_clahe = use_clahe

        self.image_col = "image"
        self.label_col = "grade"

        if self.image_col not in self.df.columns:
            raise ValueError(
                f"Column '{self.image_col}' not found in {csv_path}. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.label_col not in self.df.columns:
            raise ValueError(
                f"Column '{self.label_col}' not found in {csv_path}. "
                f"Available columns: {list(self.df.columns)}"
            )

    def _apply_clahe_rgb(self, rgb_img):
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        merged = cv2.merge([l, a, b])
        rgb_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        return rgb_img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = clean_image_name(row[self.image_col])
        label = int(row[self.label_col])

        image_path = os.path.join(self.images_dir, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        bgr = cv2.imread(image_path)

        if bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        quality_weight, quality_raw = compute_quality_score_from_gray(gray)

        if self.use_clahe:
            rgb = self._apply_clahe_rgb(rgb)

        pil_image = Image.fromarray(rgb)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label, quality_weight, quality_raw, image_name


def build_quality_split_csv(
    input_csv,
    images_dir,
    output_high_csv,
    output_low_csv,
    high_quality_ratio=0.30
):
    df = pd.read_csv(input_csv, sep=None, engine="python")
    rows = []

    if "image" not in df.columns:
        raise ValueError(
            f"Column 'image' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    for _, row in df.iterrows():
        image_name = clean_image_name(row["image"])
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue

        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        _, quality_raw = compute_quality_score_from_gray(gray)

        row_dict = row.to_dict()
        row_dict["quality_raw"] = quality_raw

        rows.append(row_dict)

    if len(rows) == 0:
        raise ValueError("No valid images found while building quality split.")

    scored_df = pd.DataFrame(rows)

    threshold = np.quantile(
        scored_df["quality_raw"].values,
        1.0 - high_quality_ratio
    )

    high_df = scored_df[scored_df["quality_raw"] >= threshold].copy()
    low_df = scored_df[scored_df["quality_raw"] < threshold].copy()

    high_df.to_csv(output_high_csv, index=False)
    low_df.to_csv(output_low_csv, index=False)

    print("===== QUALITY SPLIT CREATED =====")
    print(f"Top high-quality ratio: {high_quality_ratio}")
    print(f"Computed threshold: {threshold:.4f}")
    print(f"High-quality samples: {len(high_df)} -> {output_high_csv}")
    print(f"Low-quality samples: {len(low_df)} -> {output_low_csv}")
    print(f"Total: {len(high_df) + len(low_df)}")