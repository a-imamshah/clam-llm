import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from model import CLAMReportGenerator
from data import CLAMReportDataset

# === CONFIG ===
JSON_PATH = "train.json"
FEATURES_DIR = "/mnt/NAS_AI/ahmed/reg2025/hibou_features"
MODEL_SAVE_PATH = "clam_report_model.pt"
EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MAX_TOKENS = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Tokenizer and Dataset ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")

dataset = CLAMReportDataset(
    json_path=JSON_PATH,
    features_dir=FEATURES_DIR,
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    device=DEVICE
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Initialize Model ===
model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
model.train()
best_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0
    for i, (features, input_ids, attn_mask) in enumerate(dataloader):
        features = features[0].to(DEVICE)            # [N_patches, 1024]
        input_ids = input_ids.to(DEVICE)             # [batch, seq_len]
        attn_mask = attn_mask.to(DEVICE)

        optimizer.zero_grad()

        # Decode ground-truth text
        report_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        output = model(patch_features=features, report_text=report_text)

        loss = output.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f">>> Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {avg_loss:.4f}")

    # Save only if this is the best model so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"[✓] Best model updated (loss: {best_loss:.4f}) and saved to: {MODEL_SAVE_PATH}")
