from datasets.dataset_h5 import Whole_Slide_Bag_FP
import torch
import torch.nn as nn
import os
import time
from torch.utils.data import DataLoader
import h5py
import openslide
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login
from utils.utils import collate_features
from utils.file_utils import save_hdf5



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, print_every=20, custom_downsample=1, target_patch_size=224):

    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=True,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    _ = dataset[0]
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print(f'batch {count}/{len(loader)}, {count * batch_size} patches processed')
            batch = batch.to(device, non_blocking=True)
            features = model(batch).pooler_output.cpu().numpy()

            save_hdf5(output_path, {'features': features, 'coords': coords}, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


def extract_features_hibou(data_h5_path, slide_path, feat_dir,
                           batch_size=256, custom_downsample=1, target_patch_size=224):

    # Slide ID = filename without extension
    slide_id = os.path.splitext(os.path.basename(data_h5_path))[0]
    # Load model
    processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
    model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
    model = model.to(device).eval()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Output paths
    os.makedirs(feat_dir, exist_ok=True)
    h5_out_path = os.path.join(feat_dir, f"h5_files/{slide_id}.h5")
    pt_out_path = os.path.join(feat_dir, f"pt_files/{slide_id}.pt")  # Save .pt here

    try:
        print(f"Extracting features from:\nH5: {data_h5_path}\nSlide: {slide_path}")
        wsi = openslide.open_slide(slide_path)
        time_start = time.time()

        compute_w_loader(data_h5_path, h5_out_path, wsi, model,
                         batch_size=batch_size,
                         custom_downsample=custom_downsample,
                         target_patch_size=target_patch_size)

        elapsed = time.time() - time_start
        print(f'Finished in {elapsed:.2f}s. Saved to {h5_out_path}')

        with h5py.File(h5_out_path, "r") as f:
            features = torch.from_numpy(f['features'][:])
            print('features size:', features.shape)

        torch.save(features, pt_out_path)  # Save features tensor here
        print(f"Saved PyTorch tensor to: {pt_out_path}")

        return features  # Return features tensor to caller

    except Exception as e:
        print("Error during feature extraction:", e)
        return None

