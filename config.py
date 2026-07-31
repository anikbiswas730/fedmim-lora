"""
Central configuration for FedMIM-LoRA.

All federation, model, and data hyperparameters live here so every
module (data, model, aggregation, client, train) reads from one
source of truth.
"""

import torch

# --------------------------------------------------------------------------
# Federation
# --------------------------------------------------------------------------
NUM_CLIENTS = 10
FRACTION_FIT = 0.5          # fraction of clients sampled per round
NUM_ROUNDS = 6
LOCAL_EPOCHS = 4

# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# --------------------------------------------------------------------------
# Non-IID simulation (Dirichlet partitioning)
# --------------------------------------------------------------------------
DIRICHLET_ALPHA = 0.1       # smaller -> more extreme non-IID
NUM_CLASSES = 100           # CIFAR-100 fine labels

# --------------------------------------------------------------------------
# Masked Image Modeling
# --------------------------------------------------------------------------
MASKING_RATIO = 0.75
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# --------------------------------------------------------------------------
# LoRA (Fixed-A strategy)
# --------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES_FALLBACK = ["query", "value"]  # auto-detected in model.py
BACKBONE = "google/vit-base-patch16-224-in21k"

# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------
DEVICE_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_0)

# --------------------------------------------------------------------------
# Checkpointing / outputs
# --------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
RESUME_CHECKPOINT_PATH = ""   # set to a path to resume from a prior run
OUTPUT_DIR = "fedmim_lora_final"
