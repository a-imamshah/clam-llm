import sys
from data import CLAMReportDataset
import os
import json
import torch
from torch.utils.data import Dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CLAMReportDataset(
    json_path="/mnt/NAS_AI/ahmed/reg2025/clam-llm/train.json",
    features_dir="/mnt/NAS_AI/ahmed/reg2025/hibou_features/pt_files",
    tokenizer=None,
    max_tokens=512,
    device=DEVICE
)

print(len(dataset))
print(dataset[0])