from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_uwf import UWFDataset

TRAIN_CSV = "data/uwf_segmented/train.csv"
VAL_CSV = "data/uwf_segmented/val.csv"
TEST_CSV = "data/uwf_segmented/test.csv"
IMG_DIR = "data/uwf_segmented/all_images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = UWFDataset(TRAIN_CSV, IMG_DIR, transform)
val_dataset = UWFDataset(VAL_CSV, IMG_DIR, transform)
test_dataset = UWFDataset(TEST_CSV, IMG_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))
print("Total:", len(train_dataset) + len(val_dataset) + len(test_dataset))

images, labels = next(iter(train_loader))

print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)
print("First labels:", labels[:5].tolist())