from transformers import T5Tokenizer
from data import CLAMReportDataset

tokenizer = T5Tokenizer.from_pretrained("t5-small")
dataset = CLAMReportDataset(
    json_path='train.json',
    features_dir='/mnt/NAS_AI/ahmed/reg2025/hibou_features/pt_files',
    tokenizer=tokenizer,
    device='cuda'
)

features, report_ids, attn_mask = dataset[0]

print(features.shape)      # [N_patches, 1024]
print(report_ids.shape)    # [max_seq_len]

# import os

# feature_dir = "/mnt/NAS_AI/ahmed/reg2025/hibou_features/pt_files"
# all_feats = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])
# print("Sample .pt files:", all_feats[:10])