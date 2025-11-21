import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "data") # Assuming data is in root/data, or adjust as needed. 
# Based on requirements, data is now in classify/data
DATA_ROOT = os.path.join(PROJECT_ROOT, "data") 

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
VISUALIZATION_PLOTS_DIR = os.path.join(VISUALIZATION_DIR, "plots")
VISUALIZATION_COORDS_DIR = os.path.join(VISUALIZATION_DIR, "coordinates")

# --- Data Files ---
TRAIN_MONO_FILE = "train_mono.fasta"
TRAIN_DI_FILE = "train_di.fasta"
TEST_MONO_FILE = "test_mono.fasta"
TEST_DI_FILE = "test_di.fasta"

# --- Model Configuration ---
ESMC_MODEL_NAME = "esmc_600m" # Adjust based on actual model loading string if needed
EMBEDDING_DIM = 1152 # ESM-C 600M has d_model=1152

# --- Stage 1: Contrastive Pretraining ---
STAGE1_EPOCHS = 1000
STAGE1_BATCH_SIZE = 16 
STAGE1_LEARNING_RATE = 1e-4
LOG_INTERVAL_STEPS = 50
SAVE_CHECKPOINT_INTERVAL = 25
RESUME_FROM_CHECKPOINT = None # Path to checkpoint if resuming, e.g., os.path.join(CHECKPOINT_DIR, "stage1_epoch_5.pth")

# --- Stage 2: Classification Finetuning ---
STAGE2_EPOCHS = 50
STAGE2_BATCH_SIZE = 1
STAGE2_LEARNING_RATE = 1e-4
FREEZE_BASE_MODEL = True
UNFREEZE_LAST_N_LAYERS = 2 # 0 means all frozen if FREEZE_BASE_MODEL is True

# --- Visualization & Evaluation ---
VISUALIZATION_EPOCH_INTERVAL = 10
EVAL_EPOCH_INTERVAL = 1
VISUALIZATION_METHOD = "UMAP" # or "t-SNE"

# --- Device ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
