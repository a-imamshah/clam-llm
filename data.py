import os
import json
import torch
from torch.utils.data import Dataset

class CLAMReportDataset(Dataset):
    def __init__(self, json_path, features_dir, tokenizer=None, max_tokens=512, device='cpu'):
        self.features_dir = features_dir
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device

        with open(json_path, 'r') as f:
            self.samples = json.load(f)

        # Map existing feature files for fast lookup
        available_features = set([
            os.path.splitext(f)[0] for f in os.listdir(features_dir)
            if f.endswith('.pt')
        ])

        self.valid_samples = []
        skipped = 0

        for sample in self.samples:
            base_id = os.path.splitext(sample['id'])[0]  # removes '.tiff'
            feature_path = os.path.join(features_dir, f"{base_id}.pt")

            if base_id in available_features:
                self.valid_samples.append({
                    "feature_path": feature_path,
                    "report": sample['report']
                })
            else:
                skipped += 1

        print(f"[INFO] Loaded {len(self.valid_samples)} valid samples.")
        print(f"[INFO] Skipped {skipped} samples with missing features.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        feature_path = sample["feature_path"]
        report_text = sample["report"]

        features = torch.load(feature_path).to(self.device)  # [N_patches, 1024]

        if self.tokenizer:
            encoded = self.tokenizer(report_text,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_tokens,
                                     return_tensors='pt')
            report_ids = encoded.input_ids.squeeze(0)
            attention_mask = encoded.attention_mask.squeeze(0)
            return features, report_ids, attention_mask

        return features, report_text
