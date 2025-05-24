import os
import sys
import csv
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Ensure the project’s `src` directory is on the import path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from unet import UNet


class GeneSegDataset(Dataset):
    """
    Dataset for gene‐box segmentation. Each sample consists of an RGB image
    and its corresponding binary mask.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self.image_names = sorted(p.name for p in self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name  = self.image_names[idx]
        img_path  = self.image_dir / img_name
        mask_path = self.mask_dir  / img_name.replace(".png", "_mask.png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask  = self.transform(mask)

        return image, mask


# Hyperparameters and directory definitions
BATCH_SIZE = 4
EPOCHS     = 30
LR         = 1e-4
IMG_SIZE   = (512, 512)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now point to the “train” subdirectories
IMAGE_DIR   = BASE_DIR / "data" / "images" / "train"
MASK_DIR    = BASE_DIR / "data" / "masks"  / "genes" / "train"
MODEL_DIR   = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "data" / "results" / "training"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prepare DataLoader and model
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])
dataset = GeneSegDataset(IMAGE_DIR, MASK_DIR, transform=transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model     = UNet(n_channels=3, n_classes=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()


def compute_metrics(preds, masks, threshold=0.5):
    """
    Compute per‐batch true positives, false positives, true negatives,
    and false negatives given predicted and ground‐truth masks.
    """
    preds = (preds > threshold).float()
    masks = (masks > 0.5).float()
    tp = (preds * masks).sum().item()
    tn = ((1 - preds) * (1 - masks)).sum().item()
    fp = (preds * (1 - masks)).sum().item()
    fn = ((1 - preds) * masks).sum().item()
    return tp, fp, tn, fn


def epoch_stats(acc, dataset_size):
    """
    Aggregate accumulated counts into epoch‐level metrics:
    average loss, accuracy, precision, recall, and F1 score.
    """
    acc['loss'] /= dataset_size
    tp, fp, tn, fn = acc['tp'], acc['fp'], acc['tn'], acc['fn']
    total    = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision= tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1       = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "loss": acc['loss'],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Training loop
history   = []
best_loss = float('inf')
best_ckpt = MODEL_DIR / "best_gen_box_unet.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    acc = {"loss": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")

    for imgs, masks in loop:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds       = model(imgs)

        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running metrics
        acc['loss'] += loss.item() * imgs.size(0)
        tp, fp, tn, fn = compute_metrics(preds, masks)
        acc['tp'] += tp; acc['fp'] += fp
        acc['tn'] += tn; acc['fn'] += fn

        loop.set_postfix(loss=loss.item())

    stats = epoch_stats(acc, len(dataset))
    history.append(stats)

    # Save best model by validation loss
    if stats['loss'] < best_loss:
        best_loss = stats['loss']
        torch.save(model.state_dict(), best_ckpt)

    print(
        f" → Epoch {epoch}: "
        f"loss={stats['loss']:.4f}, "
        f"acc={stats['accuracy']:.4f}, "
        f"prec={stats['precision']:.4f}, "
        f"rec={stats['recall']:.4f}, "
        f"f1={stats['f1']:.4f}"
    )

# Persist epoch‐wise metrics to CSV
csv_path = RESULTS_DIR / "metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch","loss","accuracy","precision","recall","f1"])
    for idx, s in enumerate(history, start=1):
        writer.writerow([idx, s["loss"], s["accuracy"], s["precision"], s["recall"], s["f1"]])
print(f"\nMetrics saved to {csv_path}")

# Plot and save learning curves
epochs = list(range(1, EPOCHS + 1))
plt.figure()
for metric in ["loss","accuracy","precision","recall","f1"]:
    plt.plot(epochs, [h[metric] for h in history], label=metric)
plt.xlabel("Epoch")
plt.ylabel("Metric value")
plt.title("Training Metrics Over Epochs")
plt.legend()
plt.grid(True)
plot_path = RESULTS_DIR / "training_metrics.png"
plt.savefig(plot_path)
plt.close()
print(f"Plot saved to {plot_path}")

# Print summary of best performance
print("\n■ ■ ■ TRAINING COMPLETE ■ ■ ■")
print(f"Best checkpoint: {best_ckpt} (loss={best_loss:.4f})")
best_idx   = min(range(len(history)), key=lambda i: history[i]['loss'])
best_epoch = best_idx + 1
best_stats = history[best_idx]

best_csv = RESULTS_DIR / "best_metrics.csv"
with open(best_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch","loss","accuracy","precision","recall","f1"])
    writer.writerow([
        best_epoch,
        f"{best_stats['loss']:.6f}",
        f"{best_stats['accuracy']:.6f}",
        f"{best_stats['precision']:.6f}",
        f"{best_stats['recall']:.6f}",
        f"{best_stats['f1']:.6f}"
    ])
print(f"Best metrics (epoch {best_epoch}) saved to {best_csv}")
