## Original Directory Structure

The docker container was built from the following directory structure:

```
.../reg2025/
├── clam-llm/             
│   ├── Dockerfile
│   ├── generate_report.py
│   ├── train.py
│   ├── eval.py
│   └── ...
└── data/                 
    ├── reg2025_wsi/
    ├── reg2025_patches/
    ├── reg2025_features/
    │   ├── pt_files/
    │   └── h5_files/
    └── trained_models/
```

Where ```clam-llm/``` is the folder from which the container was built. 

```data/``` exists in the same folder as ```clam-llm/```. It contains all the folders needed for the CLAM-LLM model to run.
- ```reg2025_wsi/``` contains the raw Whole Slide Images (WSI) in .tiff format.
- ```reg2025_patches/``` contains the patches extracted from the WSIs, using the architecture borrowed from the CLAM model in .h5 format.
- ```reg2025_features/``` contains the slide-level feature vectors extracted from the slide patches in both .pt and .h5 format.
- ```trained_models/``` output directory of trained models, in .pt format.



## Container Structure and Volume Mounting

The structure of the container (after mounting) is as follows:
```
/
├── app/                                
│   ├── Dockerfile
│   ├── generate_report.py
│   ├── train.py
│   ├── eval.py
│   └── requirements.txt
│   └── ... (other files from your clam-llm project)
│
├── data/                               
│   ├── reg2025_wsi/                    
│   ├── reg2025_patches/                  
│   ├── reg2025_features/
│   │   ├── pt_files/
│   │   └── h5_files/                 
│   └── trained_models/
│
└── ...
```

Where ```app/``` is the directory of the container (equivalent to ```clam_llm/```), and ```data/``` (equivalent to ```data/```). the name of the mounted volume (```data/```) must be constant, so do the four data folders (```reg2025_wsi, reg2025_features, reg2025_patches, trained_models```). 

When mounting the volume, a file named ```data/``` is mounted, which can contain any of the four directories. The mounting process and per-script directory requirements are included in "Running Essential Processes"


## Running Essential Processes 

The image was built using the command: 
```docker build -t clam_llm .```

The default run command is as follows:
```docker run -e HF_HOME=/data/hf_cache -v <your/data/directory>:/data clam_llm```

This runs generate_report.py by default on a slide called PIT_01_00002_01.

Note: The ```<your/data/directory>:/data``` command mounts your data/ directory to the container, which should have the shape specified in the "Container Structure and Volume Mounting" section.

### generate_report.py
Requires ```trained_models``` with a .pt model inside, and ```reg2025_features/``` with the slide from which to generate the report. If the feature .pt file of the given slide is not available, uses WSI in ```reg2025_wsi``` and patches in ```reg2025_patches``` of the same slide ID to generate the slide features.

Add after default docker command:
```python generate_report.py -s <slide_ID (default PIT_01_00002_01)> -m <name_of_model_inside_trained_models/.pt (default clam_report_model_20250728_175537.pt)>```

### train.py
Requires ```reg2025_features/``` for training the model, and ```trained_models/``` for outputting the trained model.

Add after default docker command:
```python train.py -e <n_of_epochs (default 80)> -l <learning_rate (default 1e-4)>```

### eval.py
Requires ```trained_models``` with a .pt model inside, and ```reg2025_features/``` as input for model inference. Prints evaluation of the model based on the BLEU, ROUGE, KEY and EMB metrics.

Add after default docker command:
```python eval.py -m <name_of_model_inside_trained_models/.pt>```

Note: the script only works for slides in the original train.json file, which contains the ground truths.

## Note on GPU usage

Though the original script supports GPU through CUDA, the docker container, in its current state, does not. It will therefore default to using the CPU when running all scripts.

## Link to Image in Docker Hub

```<place link here>```