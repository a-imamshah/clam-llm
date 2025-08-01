import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from model import CLAMReportGenerator
from data import CLAMReportDataset
import datetime
import matplotlib.pyplot as plt
import argparse # Import argparse

# === CONFIG ===
JSON_PATH = "train.json"
FEATURES_DIR = "../data/reg2025_features/pt_files"

# MODEL_SAVE_PATH and GRAPH_SAVE_PATH will be generated with a timestamp
# This can remain outside main as it uses datetime.datetime.now()
current_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"../data/trained_models/clam_report_model_{current_timestamp_str}.pt"
GRAPH_SAVE_PATH = f"../data/trained_models/clam_report_graph_{current_timestamp_str}.png" 

# EPOCHS and LEARNING_RATE will now be parsed from arguments
BATCH_SIZE = 1
MAX_TOKENS = 512

print(f"GPU AVAILABILITY: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Main Execution Block ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CLAM-LLM model.')
    parser.add_argument('-e', '--epochs', type=int, default=80,
                        help='Number of training epochs.')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    
    args = parser.parse_args()

    # Use arguments for EPOCHS and LEARNING_RATE
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate

    print(f"[INFO] Training for {EPOCHS} epochs with learning rate {LEARNING_RATE}")

    # === Load Tokenizer and Dataset ===
    print("[INFO] Loading tokenizer and dataset...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    dataset = CLAMReportDataset(
        json_path=JSON_PATH,
        features_dir=FEATURES_DIR,
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        device=DEVICE
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[INFO] Dataset loaded with {len(dataset)} samples.")


    # === Initialize Model ===
    print("[INFO] Initializing model and optimizer...")
    model = CLAMReportGenerator(t5_model_name="t5-small").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === Training Loop ===
    print("[INFO] Starting training loop...")
    model.train()
    best_loss = float('inf')
    history_loss = [] 

    for epoch in range(EPOCHS): # Use parsed EPOCHS
        total_loss = 0
        for i, (features, input_ids, attn_mask) in enumerate(dataloader):
            # print(f"step:{i}") # Removed print per step for cleaner output
            features = features[0].to(DEVICE) # [N_patches, 1024]
            input_ids = input_ids.to(DEVICE) # [batch, seq_len]
            attn_mask = attn_mask.to(DEVICE)

            optimizer.zero_grad()

            # Decode ground-truth text
            report_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            output = model(patch_features=features, report_text=report_text)

            loss = output.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0: # Print less frequently for large datasets
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        history_loss.append(avg_loss)
        print(f">>> Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {avg_loss:.4f}")

        # Save only if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[✓] Best model updated (loss: {best_loss:.4f}) and saved to: {MODEL_SAVE_PATH}")

    # === Plotting Training History ===
    print("[INFO] Plotting training history...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), history_loss, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.xticks(range(1, EPOCHS + 1))
    plt.tight_layout()
    plt.savefig(GRAPH_SAVE_PATH)
    plt.close()

    print(f"[✓] Training loss graph saved to: {GRAPH_SAVE_PATH}")
    print("[INFO] Training process completed.")