import os
import json
import torch
import sys 
from tqdm import tqdm
from model import CLAMReportGenerator # Assuming 'model' module contains CLAMReportGenerator
from data import CLAMReportDataset   # Assuming 'data' module contains CLAMReportDataset
from transformers import T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import argparse # Import argparse

# === CONFIG ===
JSON_PATH = "train.json"
FEATURES_DIR = "../data/reg2025_features/pt_files"
# MODEL_PATH will now be determined by parser argument
TRAINED_MODELS_DIR = "../data/trained_models" # Base directory where models are stored

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Clinical Keywords for KEY score ===
clinical_keywords = ["carcinoma", "mitoses", "invasive", "grade", "ductal", "lobular", "tumor", "necrosis"]

def compute_key_score(gt, pred):
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    hits = sum(1 for kw in clinical_keywords if kw in gt_words and kw in pred_words)
    total = len([kw for kw in clinical_keywords if kw in gt_words])
    return hits / total if total > 0 else 0

# === Main Execution Block ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CLAM-LLM model performance.')
    parser.add_argument('-m', '--model_name', 
                        help='Name of the model .pt file inside the trained_models/ directory', 
                        default="clam_report_model_20250728_175537.pt")
    args = parser.parse_args()

    # Construct the full model path using the parsed argument
    MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, args.model_name)
    
    # === Load Model + Tokenizer ===
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please check the path and filename.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model.eval()

    # === Load Dataset ===
    print(f"[INFO] Loading dataset from: {JSON_PATH} with features from: {FEATURES_DIR}")
    dataset = CLAMReportDataset(
        json_path=JSON_PATH,
        features_dir=FEATURES_DIR,
        tokenizer=None, # Tokenizer is not used by dataset, only model for generation
        device=DEVICE
    )
    if len(dataset) == 0:
        print("Warning: Dataset is empty. No evaluations can be performed.")
        sys.exit(0)

    # === Load Embedding Model ===
    print("[INFO] Loading SentenceTransformer for EMB score...")
    emb_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    # === Evaluation Loop ===
    rouge_metric = Rouge()
    bleu_scores, rouge_scores, key_scores, emb_scores = [], [], [], []

    print("[INFO] Starting evaluation loop...")
    for i in tqdm(range(len(dataset))):
        features, gt_report = dataset[i]
        with torch.no_grad():
            gen_report = model.generate(features)

        # BLEU
        bleu = sentence_bleu([gt_report.split()], gen_report.split())
        bleu_scores.append(bleu)

        # ROUGE
        try:
            rouge = rouge_metric.get_scores(gen_report, gt_report)[0]['rouge-l']['f']
        except ValueError: # Handle cases where ROUGE might fail (e.g., empty strings)
            rouge = 0
        rouge_scores.append(rouge)

        # KEY
        key = compute_key_score(gt_report, gen_report)
        key_scores.append(key)

        # EMB
        emb_gt = emb_model.encode(gt_report, convert_to_tensor=True)
        emb_pred = emb_model.encode(gen_report, convert_to_tensor=True)
        emb_score = util.cos_sim(emb_gt, emb_pred).item()
        emb_scores.append(emb_score)

    # === Final Metrics ===
    if bleu_scores: # Ensure scores list is not empty before calculating average
        bleu_avg = sum(bleu_scores) / len(bleu_scores)
        rouge_avg = sum(rouge_scores) / len(rouge_scores)
        key_avg = sum(key_scores) / len(key_scores)
        emb_avg = sum(emb_scores) / len(emb_scores)

        ranking_score = 0.15 * (bleu_avg + rouge_avg) + 0.4 * key_avg + 0.3 * emb_avg

        print("\n=== Evaluation Results ===")
        print(f"BLEU:  {bleu_avg:.4f}")
        print(f"ROUGE: {rouge_avg:.4f}")
        print(f"KEY:   {key_avg:.4f}")
        print(f"EMB:   {emb_avg:.4f}")
        print(f"RANKING SCORE: {ranking_score:.4f}")
    else:
        print("\nNo evaluations were performed (dataset might be empty).")