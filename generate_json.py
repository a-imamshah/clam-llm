import torch
import os
import json
from tqdm import tqdm 
from model import CLAMReportGenerator
from transformers import T5Tokenizer


FEATURES_DIR = "../feat_test1/pt_files"
MODEL_PATH = "../data/trained_models/clam_report_model_20250728_175537.pt"
OUTPUT_JSON_FILE = "../data/predictions.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Loading tokenizer and model...")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")

all_reports_data = []

print(f"Collecting feature files from: {FEATURES_DIR}")
if not os.path.exists(FEATURES_DIR):
    raise FileNotFoundError(f"Features directory not found at: {FEATURES_DIR}")


all_filenames = os.listdir(FEATURES_DIR)
files_to_process = sorted([f for f in all_filenames if f.endswith(".pt")])#[:10]


print(f"Processing {len(files_to_process)} feature files...")


for filename in tqdm(files_to_process, desc="Generating Reports", unit="file"):


    feature_id_pt = filename
    feature_id_tiff = feature_id_pt.replace(".pt", ".tiff")
    feature_filepath = os.path.join(FEATURES_DIR, filename)


    try:
        patch_features = torch.load(feature_filepath, map_location=DEVICE).to(DEVICE)

        # Generate the report
        with torch.no_grad():
            report_text = model.generate(patch_features)

        all_reports_data.append({
            "id": feature_id_tiff,
            "report": report_text
        })


    except Exception as e:
        tqdm.write(f"  Error processing {filename}: {e}")

print("\nAll processed feature files.")

try:
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(all_reports_data, f, indent=2) # indent=2 makes the JSON human-readable
    print(f"Generated reports saved to {OUTPUT_JSON_FILE}")
except Exception as e:
    print(f"Error saving reports to JSON file: {e}")