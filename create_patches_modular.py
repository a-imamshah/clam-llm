import os
import numpy as np
from tqdm import tqdm
import h5py
from openslide import OpenSlide
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG DEFAULTS ===
DEFAULT_PATCH_SIZE = 224
DEFAULT_STEP_SIZE = 224
DEFAULT_BLOCK_SIZE = 8192
DEFAULT_TISSUE_THRESHOLD = 0.05
DEFAULT_PATCH_LEVEL = 0
DEFAULT_NUM_WORKERS = 4


def save_h5(patches, coords, save_path):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('imgs', data=patches, compression="gzip")
        f.create_dataset('coords', data=coords, compression="gzip")


def is_tissue(img_rgb, threshold):
    """Rough tissue check using grayscale + Otsu"""
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = 255 - mask
    return np.count_nonzero(tissue_mask) / tissue_mask.size > threshold


def patch_wsi_tissue_only(wsi_path, save_dir, 
                          level=DEFAULT_PATCH_LEVEL,
                          block_size=DEFAULT_BLOCK_SIZE,
                          patch_size=DEFAULT_PATCH_SIZE,
                          step_size=DEFAULT_STEP_SIZE,
                          tissue_threshold=DEFAULT_TISSUE_THRESHOLD):
    
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    save_path = os.path.join(save_dir, f"{slide_id}.h5")

    # Skip if already processed
    if os.path.exists(save_path):
        return f"[SKIPPED] {slide_id} — already exists."

    try:
        slide = OpenSlide(wsi_path)
        w, h = slide.level_dimensions[level]
    except Exception as e:
        return f"[ERROR] Cannot process {slide_id}: {e}"

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
            if not is_tissue(block_np, threshold=tissue_threshold):
                continue

            for y in range(0, bh - patch_size + 1, step_size):
                for x in range(0, bw - patch_size + 1, step_size):
                    patch = block_np[y:y+patch_size, x:x+patch_size]
                    if not is_tissue(patch, threshold=tissue_threshold):
                        continue
                    patches.append(patch)
                    coords.append((x0 + x, y0 + y))

    if not patches:
        return f"[WARNING] No valid patches in {slide_id}"

    try:
        save_h5(np.stack(patches), np.array(coords), save_path)
        return f"[SAVED] {slide_id} in {save_path} — {len(patches)} patches."
    except Exception as e:
        return f"[ERROR] Failed to save {slide_id}: {e}"


def extract_patches(wsi_path_list, save_dir,
                    num_workers=DEFAULT_NUM_WORKERS,
                    level=DEFAULT_PATCH_LEVEL,
                    block_size=DEFAULT_BLOCK_SIZE,
                    patch_size=DEFAULT_PATCH_SIZE,
                    step_size=DEFAULT_STEP_SIZE,
                    tissue_threshold=DEFAULT_TISSUE_THRESHOLD):
    """Main callable function for external use"""

    os.makedirs(save_dir, exist_ok=True)
    if isinstance(wsi_path_list, str):
        wsi_path_list = [wsi_path_list]

    print(f"[INFO] Found {len(wsi_path_list)} WSIs. Using {num_workers} threads.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(patch_wsi_tissue_only, f, save_dir, level, block_size, patch_size, step_size, tissue_threshold)
            for f in wsi_path_list
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            print(result)


# Optional CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patch extractor for WSIs.")

    parser.add_argument(
    "--wsi_dir",
    default="/mnt/NAS_AI/ahmed/reg2025/reg2025_wsi",
    help="Path to input WSI directory"
    )
    parser.add_argument(
        "--save_dir",
        default="/mnt/NAS_AI/ahmed/reg2025/reg2025_features/",
        help="Path to save patch HDF5 files"
    )

    args = parser.parse_args()

    wsi_files = [os.path.join(args.wsi_dir, f) for f in os.listdir(args.wsi_dir)
                 if f.lower().endswith(('.tiff', '.tif', '.svs'))]

    extract_patches(wsi_files, args.save_dir)
