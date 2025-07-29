
# CLAM-LLM: Whole Slide Image Report Generation using CLAM and T5

This repository provides a pipeline for generating diagnostic reports from Whole Slide Images (WSIs) using a two-stage approach:
1. **Feature extraction** using the Hibou-L Vision Transformer.
2. **Report generation** using a CLAM-based encoder and T5 decoder.

---

## 🧪 Virasoft @ Reg2025 Challenge

This work is part of Virasoft Corporation’s participation in the [Reg2025 challenge](https://codalab.lisn.upsaclay.fr/competitions/19002), focused on automated report generation from pathology slides.

---

## 📁 Project Structure

```
clam-llm/
├── datasets/
├── models/
├── trained_models/
├── utils/
├── generate_report.py         # Main script to generate a report from WSI
├── create_patches_modular.py # Patch extraction from WSI
├── extract_features_modular.py # Feature extraction using Hibou-L
└── README.md
```

---

## 🧰 Setup

```bash
git clone https://github.com/a-imamshah/clam-llm.git
cd clam-llm
pip install -r requirements.txt
```

### Additional Setup
- Install `openslide` (required for WSI reading):
  ```bash
  sudo apt install openslide-tools
  pip install openslide-python
  ```

- Authenticate to HuggingFace Hub (for Hibou-L model):
  ```bash
  huggingface-cli login
  ```

---

## 📌 Usage

### Step 1: Patch Extraction from WSI

```python
from create_patches_modular import extract_patches

extract_patches(
    wsi_path="path/to/slide.tiff",
    save_dir="./hibou_patches"
)
```

### Step 2: Feature Extraction (Hibou-L)

```python
from extract_features_modular import extract_features_hibou

extract_features_hibou(
    data_h5_path="./hibou_patches/ABC123.h5",
    slide_path="./wsi/ABC123.tiff",
    feat_dir="./hibou_features"
)
```

This will save features as:
- `./hibou_features/h5_files/ABC123.h5`
- `./hibou_features/pt_files/ABC123.pt`


### Step 3: Training

Training is handled via `train.py`



### Generate Report

```bash
python generate_report.py --slide_ID ABC123
```

This takes WSI path as argument, loads the trained model, creates patches extracts the features, and prints a T5-generated report.

---

## 🧠 Model Architecture

- **Encoder**: Modified CLAM (attention-based MIL) that encodes extracted patch features.
- **Decoder**: Pretrained `t5-small` model fine-tuned to generate pathology reports.


---

## 🧑‍💼 Maintainers

This repository is maintained by **Virasoft Corporation** as part of Reg2025 Challenge participation.

