import os
import numpy as np
from tqdm import tqdm
import h5py
from openslide import OpenSlide
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
PATCH_SIZE = 224
STEP_SIZE = 224
BLOCK_SIZE = 8192
TISSUE_THRESHOLD = 0.05
PATCH_LEVEL = 0
NUM_WORKERS = 4

WSI_DIR = '/mnt/NAS_AI/ahmed/reg2025/reg2025_wsi'
SAVE_DIR = '/mnt/NAS_AI/ahmed/reg2025/reg2025_features/'
os.makedirs(SAVE_DIR, exist_ok=True)

def save_h5(patches, coords, save_path):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('imgs', data=patches, compression="gzip")
        f.create_dataset('coords', data=coords, compression="gzip")

def is_tissue(img_rgb, threshold=TISSUE_THRESHOLD):
    """Rough tissue check using grayscale + Otsu"""
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = 255 - mask
    return np.count_nonzero(tissue_mask) / tissue_mask.size > threshold

def patch_wsi_tissue_only(wsi_path, save_dir, level=0, block_size=BLOCK_SIZE, patch_size=PATCH_SIZE, step_size=STEP_SIZE):
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    save_path = os.path.join(save_dir, f"{slide_id}.h5")

    # Skip if already processed
    if os.path.exists(save_path):
        return f"[SKIPPED] {slide_id} — already exists."

    try:
        slide = OpenSlide(wsi_path)
    except Exception as e:
        return f"[ERROR] Cannot open {slide_id}: {e}"

    try:
        w, h = slide.level_dimensions[level]
    except Exception as e:
        return f"[ERROR] Invalid level {level} in {slide_id}: {e}"

    patches, coords = [], []

    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            bw = min(block_size, w - x0)
            bh = min(block_size, h - y0)
            if bw != bh:
                continue  # skip non-square blocks

            try:
                region = slide.read_region((x0, y0), level, (bw, bh)).convert("RGB")
            except:
                continue

            block_np = np.array(region)
            if not is_tissue(block_np):
                continue

            for y in range(0, bh - patch_size + 1, step_size):
                for x in range(0, bw - patch_size + 1, step_size):
                    patch = block_np[y:y+patch_size, x:x+patch_size]
                    if not is_tissue(patch):
                        continue
                    patches.append(patch)
                    coords.append((x0 + x, y0 + y))

    if not patches:
        return f"[WARNING] No valid patches in {slide_id}"

    try:
        save_h5(np.stack(patches), np.array(coords), save_path)
        return f"[SAVED] {slide_id} — {len(patches)} patches."
    except Exception as e:
        return f"[ERROR] Failed to save {slide_id}: {e}"

def main():
    wsi_files = [os.path.join(WSI_DIR, f) for f in os.listdir(WSI_DIR)
                 if f.lower().endswith(('.tiff', '.tif', '.svs'))]

    print(f"[INFO] Found {len(wsi_files)} WSIs. Using {NUM_WORKERS} threads.")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(patch_wsi_tissue_only, f, SAVE_DIR) for f in wsi_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            print(result)

if __name__ == "__main__":
    main()