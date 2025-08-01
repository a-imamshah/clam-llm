import os
import sys
import torch
from model import CLAMReportGenerator
from transformers import T5Tokenizer
from create_patches_modular import extract_patches
from extract_features_modular import extract_features_hibou
import argparse

# === CONFIG ===
WSI_PATH = "../data/reg2025_wsi"  # folder with .tiff
PATCH_SAVE_DIR = "../data/reg2025_patches"
FEATURE_SAVE_DIR = "../data/reg2025_features"
# MODEL_PATH will now be determined by parser argument
TRAINED_MODELS_DIR = "../data/trained_models" # Base directory where models are stored

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Helper Function ===
def ensure_features(slide_id):
    feature_file = os.path.join(FEATURE_SAVE_DIR, f"pt_files/{slide_id}.pt")
    
    if os.path.exists(feature_file):
        print(f"[INFO] Features found in {feature_file}")
        return feature_file

    print(f"[INFO] Features not found for the path: {feature_file}. Creating for slide: {slide_id}")

    # Step 1: Extract patches (assumes you have a function for this)
    wsi_file = os.path.join(WSI_PATH, f"{slide_id}.tiff")
    os.makedirs(PATCH_SAVE_DIR, exist_ok=True)
    extract_patches(wsi_file, PATCH_SAVE_DIR)

    # Step 2: Extract features with Hibou-L
    extract_features_hibou(
    data_h5_path=os.path.join(PATCH_SAVE_DIR, f"{slide_id}.h5"),
    slide_path=wsi_file,
    feat_dir=FEATURE_SAVE_DIR,
    )

    print(f"[INFO] Successfully saved features for {slide_id} in {feature_file}")
    return feature_file

# === Main Entry ===
# Now accepts the 'model' object as an argument
def generate_report(slide_id, model):
    feature_path = ensure_features(slide_id)
    patch_features = torch.load(feature_path, map_location=DEVICE).to(DEVICE)

    with torch.no_grad():
        report = model.generate(patch_features)

    print("\n=== GENERATED REPORT ===")
    print(report)
    return report

parser = argparse.ArgumentParser(description='Generate a report, given a Whole Slide Image')
parser.add_argument('-s','--slide_ID', help='The ID (name) of the WSI', default="PIT_01_00002_01")
# Add a new argument for the model name
parser.add_argument('-m', '--model_name', help='Name of the model .pt file inside the trained_models/ directory', default="clam_report_model_20250728_175537.pt")


if __name__ == "__main__":
    args = parser.parse_args()

    # Define MODEL_PATH using the parsed argument
    MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, args.model_name)

    # Initialize tokenizer and model here, after MODEL_PATH is determined
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # The 'model' variable needs to be local to this scope now
    loaded_model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)
    loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    loaded_model.eval()

    slide_id = args.slide_ID.replace(".tiff", "")
    print(f"[INFO] Generatng report for the slide: {slide_id}")
    
    # Pass the loaded_model instance to the generate_report function
    generate_report(slide_id, loaded_model)