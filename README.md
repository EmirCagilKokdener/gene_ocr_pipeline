
# 🧬 Gene OCR Pipeline

A fully automated image-to-network pipeline for extracting gene labels from KEGG pathway diagrams, matching them to KGML entries, and constructing/visualizing the gene interaction graph. Includes segmentation model training, standalone inference, full pipeline execution, and optional evaluation.

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
   - [Pipeline + Evaluation](#pipeline--evaluation)
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
│   │   ├── train/
│   │   └── test/
│   ├── xml/
│   │   ├── train/
│   │   └── test/
│   ├── masks/genes/
│   └── results/
│       ├── inference/
│       └── pipeline/
├── models/
├── src/
│   ├── download_kegg.py
│   ├── generate_gene_masks_from_xml.py
│   ├── segment_genes.py
│   ├── ocr_genes.py
│   ├── match_with_kgml.py
│   ├── graph_constructor.py
│   ├── visualize_graph.py
│   ├── evaluate.py
│   └── main.py
├── train/
│   └── train_gen_box_unet.py
├── inference.py
└── requirements.txt
```

---

## 🧠 Description

This repository supports four workflows:

1. **Segmentation & Inference**
2. **Full Pipeline (EasyOCR only)**
3. **Training the Segmentation Model**
4. **Pipeline + Evaluation** using `--eval` flag

---

## 📦 Requirements

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

```bash
python src/download_kegg.py
python src/generate_gene_masks_from_xml.py
```

---

## 🚀 Usage

### 🧭 Segmentation & Inference

```bash
python inference.py
```

---

### 🔄 Full Pipeline (EasyOCR only)

```bash
python src/main.py --pathway hsa04110
```

---

### 🧪 Pipeline + Evaluation

Run full pipeline **with automatic evaluation**:
```bash
python src/main.py --pathway hsa04110 --eval
```

This evaluates precision, recall, and accuracy by comparing predicted gene entries to KGML ground truth.

---

### 🏋️ Training the Segmentation Model

```bash
python train/train_gen_box_unet.py
```

---

## 🔍 Pipeline Steps

1. Segment gene boxes (UNet)
2. OCR via EasyOCR
3. Fuzzy match OCR to KGML entries
4. Save JSON outputs
5. Build & visualize interaction graph
6. (Optional) Evaluate predictions using KGML

---

## 📊 Results & Outputs

| Task         | Output Path                                 |
|--------------|----------------------------------------------|
| Inference    | `data/results/inference/`                    |
| Pipeline     | `data/results/pipeline/{pathway}_*.json/png` |
| Evaluation   | Printed to console when `--eval` is used     |
| Training     | `models/`, `data/results/training/`          |

---

## 🧩 Extending the Pipeline

- Add OCR engines in `ocr_genes.py`
- Modify UNet training in `train_gen_box_unet.py`
- Evaluate different pathways via batch loop in `main.py`

---

## 📄 License

This project is released under the **MIT License**.  
See the LICENSE file for more info.
