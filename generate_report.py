import os
import sys
import torch
from model import CLAMReportGenerator
from transformers import T5Tokenizer
from create_patches import extract_patches
from hibou_model import extract_features_fp

# === CONFIG ===
WSI_PATH = "/mnt/NAS_AI/ahmed/reg2025/reg2025_wsi"  # folder with .tiff
PATCH_SAVE_DIR = "/mnt/NAS_AI/ahmed/reg2025/hibou_patches"
FEATURE_SAVE_DIR = "/mnt/NAS_AI/ahmed/reg2025/hibou_features"
MODEL_PATH = "clam_report_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Initialize model + tokenizer ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Helper Function ===
def ensure_features(slide_id):
    feature_file = os.path.join(FEATURE_SAVE_DIR, f"{slide_id}.pt")
    
    if os.path.exists(feature_file):
        return feature_file

    print(f"[INFO] Features not found. Creating for slide: {slide_id}")

    # Step 1: Extract patches (assumes you have a function for this)
    wsi_file = os.path.join(WSI_PATH, f"{slide_id}.tiff")
    patch_output_dir = os.path.join(PATCH_SAVE_DIR, slide_id)
    os.makedirs(patch_output_dir, exist_ok=True)
    extract_patches(wsi_file, patch_output_dir)  # <- must be implemented

    # Step 2: Extract features with Hibou-L
    features = extract_features_hibou(patch_output_dir)  # <- must be implemented
    torch.save(features, feature_file)
    return feature_file

# === Main Entry ===
def generate_report(slide_id):
    feature_path = ensure_features(slide_id)
    patch_features = torch.load(feature_path).to(DEVICE)  # shape [N_patches, 1024]

    with torch.no_grad():
        report = model.generate(patch_features)

    print("\n=== GENERATED REPORT ===")
    print(report)
    return report

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_report.py PIT_01_00002_01")
        sys.exit(1)

    slide_id = sys.argv[1].replace(".tiff", "")
    generate_report(slide_id)
