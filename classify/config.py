import os

# --- 路径配置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "data")
# 数据实际在 classify/data 目录
DATA_ROOT = os.path.join(PROJECT_ROOT, "data") 

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# --- Stage 1 配置 ---
STAGE1_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "stage1")
STAGE1_LOG_DIR = os.path.join(STAGE1_OUTPUT_DIR, "logs")
STAGE1_CHECKPOINT_DIR = os.path.join(STAGE1_OUTPUT_DIR, "checkpoints")
STAGE1_VISUALIZATION_DIR = os.path.join(STAGE1_OUTPUT_DIR, "visualizations")
STAGE1_VISUALIZATION_PLOTS_DIR = os.path.join(STAGE1_VISUALIZATION_DIR, "plots")
STAGE1_VISUALIZATION_COORDS_DIR = os.path.join(STAGE1_VISUALIZATION_DIR, "coordinates")

# --- Stage 2 配置 ---
STAGE2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "stage2")
STAGE2_LOG_DIR = os.path.join(STAGE2_OUTPUT_DIR, "logs")
STAGE2_CHECKPOINT_DIR = os.path.join(STAGE2_OUTPUT_DIR, "checkpoints")
STAGE2_VISUALIZATION_DIR = os.path.join(STAGE2_OUTPUT_DIR, "visualizations")

# 废弃的全局路径 (保留以兼容旧代码)
LOG_DIR = STAGE1_LOG_DIR 
CHECKPOINT_DIR = STAGE1_CHECKPOINT_DIR
VISUALIZATION_DIR = STAGE1_VISUALIZATION_DIR
VISUALIZATION_PLOTS_DIR = STAGE1_VISUALIZATION_PLOTS_DIR
VISUALIZATION_COORDS_DIR = STAGE1_VISUALIZATION_COORDS_DIR

# --- 数据文件 ---
TRAIN_MONO_FILE = "train_mono.fasta"
TRAIN_DI_FILE = "train_di.fasta"
TEST_MONO_FILE = "test_mono.fasta"
TEST_DI_FILE = "test_di.fasta"

# --- 模型配置 ---
ESMC_MODEL_NAME = "esmc_600m"
EMBEDDING_DIM = 1152  # ESM-C 600M 的 d_model 维度

# --- 设备配置 ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Stage 专用设备 (可设为 "cuda:0", "cuda:1" 等)
STAGE1_DEVICE = "cuda:1" 
STAGE2_DEVICE = "cuda:3"

# --- Stage 1: 对比预训练 ---
STAGE1_EPOCHS = 10000
STAGE1_BATCH_SIZE = 16 
STAGE1_LEARNING_RATE = 1e-5
LOG_INTERVAL_STEPS = 50
SAVE_CHECKPOINT_INTERVAL = 250
RESUME_FROM_CHECKPOINT = None  # 恢复训练的检查点路径
STAGE1_FULL_FINETUNE = False  # 是否解冻所有层进行全量微调

# --- Stage 2: 分类微调 ---
STAGE2_EPOCHS = 10000
STAGE2_BATCH_SIZE = 32  # 全批次 (总样本约 28 条)
STAGE2_LEARNING_RATE = 5e-6
FREEZE_BASE_MODEL = True
UNFREEZE_LAST_N_LAYERS = 0

# --- 数据增强 ---
AUGMENT_PROB = 0
MASK_PROB = 0       
MUTATION_PROB = 0

# Stage 2 预训练权重路径
# 设为 None 跳过加载 Stage 1 权重，或指定自定义路径
STAGE2_PRETRAINED_WEIGHTS = "/home/wangty/esm/esm/classify/output/stage1_20251128_final/checkpoints/stage1_epoch_1000.pth"

# Stage 2 检查点保存间隔
STAGE2_SAVE_CHECKPOINT_INTERVAL = 25

# --- 可视化与评估 ---
# Stage 1
STAGE1_VISUALIZATION_INTERVAL = 10  # 可视化生成间隔
STAGE1_SAVE_PLOTS_INTERVAL = 1      # 训练曲线保存间隔

# Stage 2
STAGE2_VISUALIZATION_INTERVAL = 10  # 可视化生成间隔
STAGE2_EVAL_INTERVAL = 10           # 测试集评估间隔
STAGE2_SAVE_PLOTS_INTERVAL = 1      # 训练曲线保存间隔

# 废弃 (保留兼容性)
VISUALIZATION_EPOCH_INTERVAL = STAGE1_VISUALIZATION_INTERVAL
EVAL_EPOCH_INTERVAL = STAGE2_EVAL_INTERVAL

VISUALIZATION_METHOD = "UMAP"  # 或 "t-SNE"

# --- 预测配置 ---
# 预测使用的模型权重路径
PREDICT_MODEL_PATH = "/home/wangty/esm/esm/classify/output/stage2_20251128_final/checkpoints/stage2_epoch_1200.pth"

# 学习率预热比例
WARMUP_RATIO = 0.1
