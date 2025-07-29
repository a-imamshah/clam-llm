import os
import sys
import torch
from model import CLAMReportGenerator
from transformers import T5Tokenizer
from create_patches_modular import extract_patches
from extract_features_modular import extract_features_hibou
import argparse

# === CONFIG ===
WSI_PATH = "/mnt/NAS_AI/ahmed/reg2025/reg2025_wsi"  # folder with .tiff
PATCH_SAVE_DIR = "/mnt/NAS_AI/ahmed/reg2025/hibou_patches"
FEATURE_SAVE_DIR = "/mnt/NAS_AI/ahmed/reg2025/hibou_features"
MODEL_PATH = "../trained_models/clam_report_model_20250728_175537.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Initialize model + tokenizer ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Helper Function ===
def ensure_features(slide_id):
    feature_file = os.path.join(FEATURE_SAVE_DIR, f"pt_files/{slide_id}.pt")
    
    if os.path.exists(feature_file):
        print(f"[INFO] Features found in {feature_file}")
        return feature_file

    print(f"[INFO] Features not found for the path: {feature_file}. Creating for slide: {slide_id}")

    # Step 1: Extract patches (assumes you have a function for this)
    wsi_file = os.path.join(WSI_PATH, f"{slide_id}.tiff")
            # patch_output_dir = os.path.join(PATCH_SAVE_DIR, slide_id)
    os.makedirs(PATCH_SAVE_DIR, exist_ok=True)
    extract_patches(wsi_file, PATCH_SAVE_DIR)  # creates .h5 patches from wsi, and saves it to patch_output_dir

    # Step 2: Extract features with Hibou-L
    
    extract_features_hibou(
    data_h5_path=os.path.join(PATCH_SAVE_DIR, f"{slide_id}.h5"),
    slide_path=wsi_file,
    feat_dir=FEATURE_SAVE_DIR,
    )

    print(f"[INFO] Successfully saved features for {slide_id} in {feature_file}")
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

parser = argparse.ArgumentParser(description='Generate a report, given a Whole Slide Image')
parser.add_argument('-s','--slide_ID', help='The ID (name) of the WSI', default="PIT_01_00002_01")

if __name__ == "__main__":
    args = parser.parse_args()

    # if len(sys.argv) != 2:
    #     print("Usage: python generate_report.py PIT_01_00002_01")
    #     sys.exit(1)

    #slide_id = sys.argv[1].replace(".tiff", "")

    slide_id = args.slide_ID.replace(".tiff", "")
    print(f"[INFO] Generatng report for the slide: {slide_id}")
    generate_report(slide_id)
