from torchvision import transforms
from dataset_uwf import UWFDataset

CSV_PATH = "data/uwf_segmented/train.csv"
IMG_DIR = "data/uwf_segmented/all_images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = UWFDataset(
    csv_path=CSV_PATH,
    images_dir=IMG_DIR,
    transform=transform
)

print("Total samples:", len(dataset))

img, label = dataset[0]

print("Image tensor shape:", img.shape)
print("Label:", label)