# src/inference.py

import os
import json
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from unet import UNet

# 1) Define paths
BASE_DIR   = Path(__file__).resolve().parent.parent
IMG_DIR    = BASE_DIR / "data" / "images"
GENE_MODEL = BASE_DIR / "models" / "best_gen_box_unet.pth"

OUT_DIR    = BASE_DIR / "data" / "results" / "inference"
MASK_DIR   = OUT_DIR / "masks"   / "genes"
BOX_DIR    = OUT_DIR / "boxes"   / "genes"
DRAW_DIR   = OUT_DIR / "overlay"

# Ensure output directories exist
for directory in (MASK_DIR, BOX_DIR, DRAW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# 2) Model loading function
def load_unet(model_path: Path) -> UNet:
    """
    Load a UNet model from the specified checkpoint.

    Parameters
    ----------
    model_path : Path
        Filesystem path to the saved model weights.

    Returns
    -------
    model : UNet
        A UNet instance loaded with the pretrained weights.
    """
    model = UNet(n_channels=3, n_classes=1).eval()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model

# Initialize device and model
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gene_model = load_unet(GENE_MODEL).to(device)

# 3) Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 4) Utility to convert binary mask to bounding boxes
def mask_to_boxes(mask: np.ndarray, original_size: tuple) -> list:
    """
    Convert a binary segmentation mask into a list of bounding boxes.

    Parameters
    ----------
    mask : np.ndarray
        2D array of shape (H, W) with values 0 or 1.
    original_size : tuple of int
        The width and height to which the mask should be resized.

    Returns
    -------
    boxes : list of lists
        Each element is [x, y, width, height] of a detected contour.
    """
    # Resize mask back to original image dimensions
    mask_resized = cv2.resize(
        mask.astype(np.uint8) * 255,
        original_size,
        interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(
        mask_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes

# 5) Inference loop over all images
all_results = {}

for img_path in IMG_DIR.glob("*.png"):
    pathway_id = img_path.stem
    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size

    # Prepare input tensor
    tensor = preprocess(image).unsqueeze(0).to(device)

    # Perform forward pass to obtain gene probability map
    with torch.no_grad():
        output = gene_model(tensor)[0, 0].cpu().numpy()

    # Binarize predictions with a threshold of 0.5
    binary_mask = (output > 0.5).astype(np.uint8)

    # Derive bounding boxes from the segmentation mask
    gene_boxes = mask_to_boxes(binary_mask, (orig_w, orig_h))

    # Save the binary mask as an image
    cv2.imwrite(str(MASK_DIR / f"{pathway_id}_mask.png"), binary_mask * 255)

    # Save bounding boxes in JSON format
    with open(BOX_DIR / f"{pathway_id}_boxes.json", "w") as f:
        json.dump(gene_boxes, f, indent=2)

    # Generate an overlay visualization
    overlay = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for x, y, w, h in gene_boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(str(DRAW_DIR / f"{pathway_id}_overlay.png"), overlay)

    # Aggregate results for all pathways
    all_results[pathway_id] = gene_boxes

# Save all bounding boxes across pathways in a single JSON file
with open(OUT_DIR / "all_gene_boxes.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(
    "\nInference complete. Outputs:\n"
    f"- Gene masks directory: {MASK_DIR}\n"
    f"- Bounding boxes directory: {BOX_DIR}\n"
    f"- Overlay images directory: {DRAW_DIR}\n"
    f"- Consolidated boxes file: {OUT_DIR / 'all_gene_boxes.json'}"
)
