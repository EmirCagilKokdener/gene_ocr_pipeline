import json
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from unet import UNet

# 1) Define project paths
BASE_DIR   = Path(__file__).resolve().parent.parent
IMG_DIR    = BASE_DIR / "data" / "images" / "test"      # infer on held-out “test” set
MODEL_PATH = BASE_DIR / "models" / "best_gen_box_unet.pth"

OUT_DIR    = BASE_DIR / "data" / "results" / "inference"
MASK_DIR   = OUT_DIR / "masks"   / "genes"
BOX_DIR    = OUT_DIR / "boxes"   / "genes"
DRAW_DIR   = OUT_DIR / "overlay"

for d in (MASK_DIR, BOX_DIR, DRAW_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_unet(model_path: Path) -> UNet:
    """
    Instantiate a UNet model and load pretrained weights.
    """
    model = UNet(n_channels=3, n_classes=1).eval()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


# 2) Initialize device and model
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gene_model = load_unet(MODEL_PATH).to(device)


# 3) Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def mask_to_boxes(mask: np.ndarray, original_size: tuple) -> list:
    """
    Convert a binary mask into a list of bounding boxes of connected components.
    """
    mask_resized = cv2.resize(
        (mask * 255).astype(np.uint8),
        original_size,
        interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(
        mask_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes


# 5) Run inference over all test images
all_results = {}

for img_path in IMG_DIR.glob("*.png"):
    pid = img_path.stem
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    inp = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = gene_model(inp)[0, 0].cpu().numpy()

    binary_mask = (out > 0.5).astype(np.uint8)
    boxes       = mask_to_boxes(binary_mask, (orig_w, orig_h))

    # Save mask
    cv2.imwrite(str(MASK_DIR / f"{pid}_mask.png"), binary_mask * 255)

    # Save boxes JSON
    with open(BOX_DIR / f"{pid}_boxes.json", "w") as f:
        json.dump(boxes, f, indent=2)

    # Save overlay visualization
    vis = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(str(DRAW_DIR / f"{pid}_overlay.png"), vis)

    # Accumulate
    all_results[pid] = boxes

# Write consolidated JSON
with open(OUT_DIR / "all_gene_boxes.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(
    "\nInference complete. Outputs saved to:\n"
    f" • Masks:   {MASK_DIR}\n"
    f" • Boxes:   {BOX_DIR}\n"
    f" • Overlay: {DRAW_DIR}\n"
    f" • Summary: {OUT_DIR/'all_gene_boxes.json'}"
)
