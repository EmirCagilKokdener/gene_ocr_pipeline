
# src/segment_genes.py
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from unet import UNet
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_gen_box_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (512, 512)

# =============================================================================
# Model Initialization and Preprocessing Pipeline
# =============================================================================
# Instantiate the U-Net model for binary segmentation and load pretrained weights.
model = UNet(n_channels=3, n_classes=1).to(DEVICE).eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Define the image transform to resize input to the model's expected resolution.
preproc = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# =============================================================================
# Gene Segmentation Function
# =============================================================================
def segment_genes(image_path: str) -> list[tuple[int, int, int, int]]:
    """
    Perform gene region segmentation on an input pathway image.

    Steps:
      1. Load and convert the input image to RGB.
      2. Resize and normalize the image for U-Net inference.
      3. Apply the trained model to obtain a binary segmentation mask.
      4. Upsample the mask to the original image dimensions.
      5. Extract contour-based bounding boxes for regions exceeding an area threshold.
      6. Sort and return the bounding boxes by vertical and horizontal position.

    Args:
        image_path (str): Path to the input pathway image.

    Returns:
        list of tuples: Each tuple contains (x, y, width, height) for a detected gene region.
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    inp = preproc(img).unsqueeze(0).to(DEVICE)

    # Inference: generate the segmentation mask
    with torch.no_grad():
        mask_pred = model(inp)[0, 0].cpu().numpy()
    # Binarize the mask using a threshold
    bin_mask = (mask_pred > 0.5).astype(np.uint8) * 255

    # Resize mask to match the original image resolution
    bin_mask_resized = cv2.resize(
        bin_mask,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Detect contours corresponding to gene regions
    contours, _ = cv2.findContours(
        bin_mask_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter and compute bounding boxes for sufficiently large regions
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    # Sort bounding boxes in row-major order (top-to-bottom, left-to-right)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

