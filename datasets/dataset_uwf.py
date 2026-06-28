import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class UWFDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path, sep=None, engine="python")
        self.images_dir = images_dir
        self.transform = transform

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

    def _clean_image_name(self, image_name):
        """
        Converts:
            img/tr000001 -> tr000001.png
        """
        image_name = str(image_name).strip()
        image_name = image_name.split("/")[-1]

        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            image_name += ".png"

        return image_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = self._clean_image_name(row[self.image_col])
        label = int(row[self.label_col])

        image_path = os.path.join(self.images_dir, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # pretrained models need 3 channels
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, image_name