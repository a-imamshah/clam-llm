import torch
import os
import json
from tqdm import tqdm 
from model import CLAMReportGenerator
from transformers import T5Tokenizer


FEATURES_DIR = "../feat_test1/pt_files"
MODEL_PATH = "../data/trained_models/clam_report_model_20250728_175537.pt"
INPUT_JSON_FILE = "../data/virasoft_reports_test_1.json"  # New: Path to the input JSON file
OUTPUT_JSON_FILE = "../data/virasoft_reports_test_1.json" # Modified output file name

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Loading tokenizer and model...")

# T5Tokenizer and CLAMReportGenerator instantiation
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# Load the comprehensive JSON file
try:
    with open(INPUT_JSON_FILE, 'r') as f:
        all_reports_data = json.load(f)
    print(f"Successfully loaded {len(all_reports_data)} entries from {INPUT_JSON_FILE}")
except FileNotFoundError:
    print(f"Error: Input JSON file not found at: {INPUT_JSON_FILE}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {INPUT_JSON_FILE}")
    exit()

# Create a dictionary for efficient lookup
report_lookup = {item['id']: item for item in all_reports_data}

# Get list of .pt files from the features directory
print(f"Collecting feature files from: {FEATURES_DIR}")
if not os.path.exists(FEATURES_DIR):
    raise FileNotFoundError(f"Features directory not found at: {FEATURES_DIR}")

all_filenames = os.listdir(FEATURES_DIR)
files_to_process = sorted([f for f in all_filenames if f.endswith(".pt")])

# Counters for tracking the process
skipped_count = 0
modified_count = 0
not_found_in_json_count = 0

print(f"Processing {len(files_to_process)} feature files...")

# Iterate through the .pt files
for filename in tqdm(files_to_process, desc="Generating Reports", unit="file"):
    feature_id_pt = filename
    feature_id_tiff = feature_id_pt.replace(".pt", ".tiff")
    feature_filepath = os.path.join(FEATURES_DIR, filename)

    # Check if the slide ID exists in the JSON and if its report is empty
    if feature_id_tiff in report_lookup:
        if report_lookup[feature_id_tiff]['report'] == "":
            try:
                # Load features and generate report
                patch_features = torch.load(feature_filepath, map_location=DEVICE).to(DEVICE)
                with torch.no_grad():
                    report_text = model.generate(patch_features)

                # Update the report in the lookup dictionary
                report_lookup[feature_id_tiff]['report'] = report_text
                modified_count += 1
            except Exception as e:
                tqdm.write(f"  Error processing {filename}: {e}")
        else:
            # Report is not empty, so skip
            skipped_count += 1
            tqdm.write(f"  Skipping {filename}: Report already exists.")
    else:
        # .pt file ID not found in the JSON
        not_found_in_json_count += 1
        tqdm.write(f"  Warning: {filename} not found in input JSON. Skipping.")

print("\n--- Processing Summary ---")
print(f"Number of .pt files processed: {len(files_to_process)}")
print(f"Number of reports generated/modified: {modified_count}")
print(f"Number of .pt files skipped (report already exists): {skipped_count}")
print(f"Number of .pt files not found in the input JSON: {not_found_in_json_count}")
print("--------------------------")

# Save the updated JSON file
try:
    # Convert the dictionary back to a list of dictionaries for saving
    updated_reports_list = list(report_lookup.values())
    
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(updated_reports_list, f, indent=2)
    print(f"Generated and updated reports saved to {OUTPUT_JSON_FILE}")
except Exception as e:
    print(f"Error saving reports to JSON file: {e}")