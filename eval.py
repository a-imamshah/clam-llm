import os
import json
import torch
from tqdm import tqdm
from model import CLAMReportGenerator
from data import CLAMReportDataset
from transformers import T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
JSON_PATH = "train.json"
FEATURES_DIR = "/mnt/NAS_AI/ahmed/reg2025/hibou_features/pt_files"
MODEL_PATH = "clam_report_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model + Tokenizer ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Load Dataset ===
dataset = CLAMReportDataset(
    json_path=JSON_PATH,
    features_dir=FEATURES_DIR,
    tokenizer=None,
    device=DEVICE
)

# === Load Embedding Model ===
emb_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# === Define Clinical Keywords for KEY score ===
clinical_keywords = ["carcinoma", "mitoses", "invasive", "grade", "ductal", "lobular", "tumor", "necrosis"]

def compute_key_score(gt, pred):
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    hits = sum(1 for kw in clinical_keywords if kw in gt_words and kw in pred_words)
    total = len([kw for kw in clinical_keywords if kw in gt_words])
    return hits / total if total > 0 else 0

# === Evaluation Loop ===
rouge_metric = Rouge()
bleu_scores, rouge_scores, key_scores, emb_scores = [], [], [], []

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
    except:
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