
# 🧬 Gene OCR Pipeline

A fully automated image-to-network pipeline for extracting gene labels from KEGG pathway diagrams, matching them to KGML entries, and constructing/visualizing the gene interaction graph. Includes segmentation model training, standalone inference, and end-to-end pipeline.

---

## 📚 Table of Contents

1. [Project Structure](#project-structure)  
2. [Description](#description)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Data Preparation & Split](#data-preparation--split)  
6. [Usage](#usage)  
   - [Segmentation & Inference](#segmentation--inference)  
   - [Full Pipeline (EasyOCR only)](#full-pipeline-easyocr-only)  
   - [Training the Segmentation Model](#training-the-segmentation-model)  
7. [Pipeline Steps](#pipeline-steps)  
8. [Results & Outputs](#results--outputs)  
9. [Extending the Pipeline](#extending-the-pipeline)  
10. [License](#license)  

---

## 📁 Project Structure

```
.
├── data/
│   ├── images/
│   │   ├── train/                   # 10 pathways for training UNet
│   │   └── test/                    # 2 held-out pathways for evaluation
│   ├── xml/
│   │   ├── train/                   # KGML XML for training
│   │   └── test/                    # KGML XML for testing
│   ├── masks/genes/                # Auto-generated ground-truth masks
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
│   └── best_gen_box_unet.pth       # Trained UNet model
├── src/
│   ├── download_kegg.py            # Downloads and splits KEGG pathways
│   ├── generate_gene_masks_from_xml.py
│   ├── segment_genes.py            # UNet segmentation
│   ├── ocr_genes.py                # EasyOCR
│   ├── match_with_kgml.py          # Fuzzy match OCR to KGML
│   ├── graph_constructor.py        # Graph construction
│   ├── visualize_graph.py          # Graph drawing
│   └── main.py                     # End-to-end pipeline
├── train/
│   └── train_gen_box_unet.py       # UNet training script
├── inference.py                    # Standalone segmentation + box extraction
└── requirements.txt
```

---

## 🧠 Description

This repository supports three workflows:

1. **Segmentation & Inference**  
   Apply a pretrained UNet to detect gene-label boxes in pathway images, outputting masks, bounding boxes, and overlayed visuals.

2. **Full Pipeline (EasyOCR only)**  
   Segmentation → OCR (EasyOCR) → KGML matching → Graph construction → Visualization.

3. **Training**  
   Train a UNet segmentation model on XML-derived ground-truth masks.

---

## 📦 Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- OpenCV (`cv2`)  
- EasyOCR  
- NetworkX  
- matplotlib  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YourUserName/gene_ocr_pipeline.git
cd gene_ocr_pipeline
pip install -r requirements.txt
```

---

## 🗂️ Data Preparation & Split

Download KEGG pathways (10 train + 2 test):

```bash
python src/download_kegg.py
```

- Populates:  
  `data/images/train/`, `data/xml/train/`  
  `data/images/test/`, `data/xml/test/`

Generate ground-truth masks for training:

```bash
python src/generate_gene_masks_from_xml.py
```

Outputs to: `data/masks/genes/`

---

## 🚀 Usage

### 🧭 Segmentation & Inference

Run inference on all test images:

```bash
python inference.py
```

- Masks → `data/results/inference/masks/genes/`  
- Boxes (JSON) → `data/results/inference/boxes/genes/`  
- Overlays → `data/results/inference/overlay/`  
- Combined boxes → `data/results/inference/all_gene_boxes.json`

---

### 🔄 Full Pipeline (EasyOCR only)

Run the full pipeline on a test pathway:

```bash
python src/main.py --pathway hsaXXXXXX
```

- Outputs:
  - JSON: `data/results/pipeline/{pathway}_pipeline.json`
  - Graph PNG: `data/results/pipeline/{pathway}_graph.png`

---

### 🏋️ Training the Segmentation Model

Train on the train split:

```bash
python train/train_gen_box_unet.py
```

- Inputs:  
  `data/images/train/`  
  `data/masks/genes/`

- Outputs:  
  - Model → `models/best_gen_box_unet.pth`  
  - Metrics → `data/results/training/metrics.csv`  
  - Plots → `training_metrics.png`

---

## 🔍 Pipeline Steps

1. **Segmentation**:  
   Resizes image → runs UNet → thresholds → finds contours → bounding boxes.

2. **OCR (EasyOCR)**:  
   Crops + pads box → upscales → runs OCR → extracts top candidate.

3. **KGML Matching**:  
   Builds synonym→entry_id map → fuzzy match OCR to `graphics@name`.

4. **Serialization**:  
   Saves: `gene_boxes`, `gene_names`, `index_by_entry`.

5. **Graph Construction**:  
   Parses `<relation>` tags → builds NetworkX DiGraph.

6. **Visualization**:  
   Spring layout → draw nodes/edges/labels → save PNG.

---

## 📊 Results & Outputs

| Workflow   | Output Location                            |
|------------|---------------------------------------------|
| Inference  | `data/results/inference/`                   |
| Pipeline   | `data/results/pipeline/{pathway}_*.json/png`|
| Training   | `data/results/training/`, `models/`         |

Example pipeline JSON:
```json
{
  "gene_boxes": [[x, y, w, h], …],
  "gene_names": ["TP53", "CDK1", …],
  "index_by_entry": { "52": 0, "4": 1, … }
}
```

---

## 🧩 Extending the Pipeline

- Add new OCR models → `src/ocr_genes.py`
- Modify training code → `train/train_gen_box_unet.py`
- Batch evaluation → loop over `data/images/test/`

---

## 📄 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
