
# 🧬 Gene OCR Pipeline

A fully automated image-to-network pipeline for extracting gene labels from KEGG pathway diagrams, matching them to KGML entries, and constructing/visualizing the gene interaction graph. Includes segmentation model training and standalone inference.

---

## 📚 Table of Contents

1. [Project Structure](#project-structure)  
2. [Description](#description)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Data Preparation](#data-preparation)  
6. [Usage](#usage)  
7. [Pipeline Steps](#pipeline-steps)  
   - [Segmentation & Inference](#segmentation--inference)  
   - [Full Pipeline](#full-pipeline)  
   - [Training the Segmentation Model](#training-the-segmentation-model)  
8. [Results & Outputs](#results--outputs)  
9. [Extending the Pipeline](#extending-the-pipeline)  
10. [License](#license)  

---

## 📁 Project Structure

```
.
├── data/
│   ├── images/                # Pathway PNGs
│   ├── xml/                   # KGML XML files
│   ├── masks/genes/           # Ground-truth masks
│   └── results/
│       ├── inference/
│       │   ├── masks/genes/
│       │   ├── boxes/genes/
│       │   ├── overlay/
│       │   └── all_gene_boxes.json
│       └── pipeline/
│           ├── {pathway}_pipeline.json
│           └── {pathway}_graph.png
├── models/
│   └── best_gen_box_unet.pth  # Pretrained UNet model
├── src/
│   ├── download_kegg.py
│   ├── generate_gene_masks_from_xml.py
│   ├── segment_genes.py
│   ├── ocr_genes.py
│   ├── match_with_kgml.py
│   ├── graph_constructor.py
│   ├── visualize_graph.py
│   └── main.py
├── train/
│   └── train_gen_box_unet.py  # UNet training script
├── inference.py               # Standalone segmentation & box extraction
└── requirements.txt
```

---

## 🧠 Description

This repository provides:

- **Segmentation & Inference**  
  Detects gene-label boxes in pathway images using a UNet model, and saves masks, bounding boxes, and overlay images.

- **Full Pipeline**  
  Segmentation → OCR → KGML matching → Graph construction → Visualization.

- **Training**  
  Full training code for UNet using XML-derived ground-truth masks.

---

## 📦 Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- OpenCV (`cv2`)  
- EasyOCR  
- NetworkX  
- matplotlib  
- pytesseract *(optional)*

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YourUserName/gene_ocr_pipeline.git
cd gene_ocr_pipeline
pip install -r requirements.txt
```

---

## 🧪 Data Preparation

Download KEGG data:

```bash
python src/download_kegg.py
```

This will populate `data/images/` and `data/xml/`.

Optionally, generate ground-truth masks:

```bash
python src/generate_gene_masks_from_xml.py
```

Masks will be saved to `data/masks/genes/`.

---

## 🚀 Usage

### 🧭 Segmentation & Inference

Run inference on all pathway images:

```bash
python inference.py
```

- Masks → `data/results/inference/masks/genes/`  
- Boxes (JSON) → `data/results/inference/boxes/genes/`  
- Overlays → `data/results/inference/overlay/`  
- All boxes (combined) → `data/results/inference/all_gene_boxes.json`

---

### 🔄 Full Pipeline

Run the complete pipeline for a single pathway:

```bash
python src/main.py --pathway hsa04110
```

---

## 🔍 Pipeline Steps

### Segmentation & Inference

- `segment_genes.py`:  
  Loads the UNet model → resizes image → thresholds prediction → finds contours → outputs bounding boxes.

- `inference.py`:  
  Wraps segmentation + saves masks, boxes, overlays.

---

### Full Pipeline

1. **Segmentation** → `segment_genes`
2. **OCR** → `extract_gene_names` (via EasyOCR)
3. **KGML Matching** → `match_with_kgml`  
   Uses fuzzy string match to `graphics@name` entries.
4. **Save JSON**  
   Includes `gene_boxes`, `gene_names`, `index_by_entry`.
5. **Graph Construction** → `graph_constructor`
6. **Visualization** → `visualize_graph`  
   Saves high-resolution PNG.

---

### 🏋️ Training the Segmentation Model

Train the UNet using your ground-truth masks:

```bash
python train/train_gen_box_unet.py
```

- Input: `data/images/`, `data/masks/genes/`
- Output:
  - Model → `models/best_gen_box_unet.pth`
  - Logs → `data/results/training/metrics.csv`
  - Plots → `training_metrics.png`

---

## 📊 Results & Outputs

| Task         | Output Location                              |
|--------------|-----------------------------------------------|
| Inference    | `data/results/inference/`                     |
| Pipeline     | `data/results/pipeline/{pathway}_*.json/png`  |
| Training     | `data/results/training/` & `models/`          |

Example JSON:
```json
{
  "gene_boxes": [[x, y, w, h], ...],
  "gene_names": ["TP53", "CDK1", ...],
  "index_by_entry": { "52": 0, "4": 1 }
}
```

---

## 🧩 Extending the Pipeline

- Add new OCR engines → edit `src/ocr_genes.py`
- Retrain model → modify `train/train_gen_box_unet.py`
- Batch mode → iterate over all files in `data/images/`

---

## 📄 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
