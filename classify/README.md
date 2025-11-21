# ESM-C 小样本蛋白质分类工具

基于 ESM-C (600M) 和 SETFIT 策略的蛋白质序列二分类工具。

## 📋 目录

- [项目简介](#项目简介)
- [技术方案](#技术方案)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [配置参数](#配置参数)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

## 项目简介

本工具实现了一个高效、无提示(Prompt-free)的蛋白质序列二分类系统,采用两阶段训练策略:

1. **阶段 I (对比学习预训练)**: 通过对比学习微调 ESM-C 编码器,使其学习区分正负序列对的语义空间
2. **阶段 II (分类头训练)**: 在微调后的编码器基础上训练分类头,实现精准的二分类

### 核心特性

- ✅ **小样本学习**: 采用 SETFIT 框架,适合少量标注数据的场景
- ✅ **高质量可视化**: 提供 UMAP/t-SNE 降维可视化,清晰展示语义空间分布
- ✅ **完整输出**: 生成 CSV、FASTA 等多种格式的预测结果
- ✅ **实时监控**: 训练和预测过程实时输出进度和指标
- ✅ **断点恢复**: 支持训练中断后从检查点继续

## 技术方案

### 基座模型
- **ESM-C 600M**: EvolutionaryScale 的蛋白质语言模型
- **嵌入维度**: 1152

### 训练策略
1. **阶段 I**: 对比学习 (CosineEmbeddingLoss)
   - 正样本对: 来自同一类别的序列对 (target=1)
   - 负样本对: 来自不同类别的序列对 (target=-1)
   
2. **阶段 II**: 分类微调 (CrossEntropyLoss)
   - 可选择冻结/解冻编码器层
   - 训练简单的 MLP 分类头

### 可视化方法
- **降维算法**: UMAP 或 t-SNE
- **输出时机**: 每个阶段周期性生成可视化图片
- **展示内容**: 不同类别序列在语义空间中的分布

## 项目结构

```
classify/
├── config.py              # ⚙️ 配置参数 (所有可调参数集中在此)
├── data_loader.py         # 💾 数据加载与样本生成
├── model.py              # 🧠 模型定义 (ESM-C + 分类头)
├── utils.py              # 🛠️ 工具函数 (日志、检查点、可视化)
├── stage1_pretrain.py    # 🚀 阶段 I: 对比学习预训练
├── stage2_finetune.py    # 🚀 阶段 II: 分类头训练
├── predict.py            # 🔮 推理预测脚本
├── README.md             # 📖 本文档
├── data/                 # 📁 数据目录
│   ├── train_mono.fasta  # 训练集 - 类别 1
│   ├── train_di.fasta    # 训练集 - 类别 2
│   ├── test_mono.fasta   # 测试集 - 类别 1
│   └── test_di.fasta     # 测试集 - 类别 2
└── output/               # 📂 输出目录
    ├── logs/             # 日志文件
    │   ├── stage1_log.txt
    │   ├── stage2_log.txt
    │   └── predict_log.txt
    ├── checkpoints/      # 模型检查点
    │   ├── stage1_epoch_*.pth
    │   ├── stage1_final.pth
    │   └── stage2_final.pth
    └── visualizations/   # 可视化图片
        ├── stage1_epoch_*.png
        ├── stage2_epoch_*.png
        └── prediction_vis.png
```

## 环境配置

### 服务器环境

本项目需要在配置了 ESM-C 模型的服务器上运行:

```bash
# 1. SSH 连接到服务器
ssh -t wangty@10.19.147.98

# 2. 激活 esm3 环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate esm3

# 3. 进入项目目录
cd esm/esm/classify
```

### 依赖包

确保以下 Python 包已安装:

```bash
# 核心依赖
torch
numpy

# 可视化依赖
matplotlib
umap-learn
scikit-learn

# ESM 相关
esm (已在 esm3 环境中)
```

如果缺少依赖,可以安装:

```bash
python3 -m pip install umap-learn matplotlib scikit-learn
```

## 数据准备

### 数据格式

所有数据文件使用标准 FASTA 格式:

```fasta
>sequence_id_1
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>sequence_id_2
KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE
```

### 数据文件

将您的数据文件放在 `classify/data/` 目录下:

| 文件名 | 说明 | 用途 |
|--------|------|------|
| `train_mono.fasta` | 训练集 - 类别 1 (例如:单铜蛋白) | 阶段 I/II 训练 |
| `train_di.fasta` | 训练集 - 类别 2 (例如:双铜蛋白) | 阶段 I/II 训练 |
| `test_mono.fasta` | 测试集 - 类别 1 | 阶段 I/II 评估与可视化 |
| `test_di.fasta` | 测试集 - 类别 2 | 阶段 I/II 评估与可视化 |

**注意**: 
- 类别 1 对应标签 0 (Mono)
- 类别 2 对应标签 1 (Di)

## 使用方法

### 完整训练流程

#### 1. 阶段 I: 对比学习预训练

```bash
python3 stage1_pretrain.py
```

**输出示例**:
```
[2025-11-20 21:59:19] Starting Stage 1: Contrastive Pretraining
[2025-11-20 21:59:19] Using device: cuda
[2025-11-20 21:59:19] Initializing model...
Loading ESM-C model...
ESM-C model loaded.
[2025-11-20 21:59:31] Model initialized and moved to device.
  Epoch 1, Step 10/100, Loss: 0.1234
  Epoch 1, Step 20/100, Loss: 0.0987
--- Epoch 1 Complete. Avg Train Loss: 0.1042 ---
[2025-11-20 21:59:34] Generating visualization...
Visualization saved to output/visualizations/stage1_epoch_1.png
Checkpoint saved to output/checkpoints/stage1_epoch_1.pth
```

**生成文件**:
- `output/logs/stage1_log.txt` - 训练日志
- `output/checkpoints/stage1_final.pth` - 最终模型
- `output/visualizations/stage1_epoch_*.png` - 可视化图片

#### 2. 阶段 II: 分类头训练

```bash
python3 stage2_finetune.py
```

**输出示例**:
```
[2025-11-20 22:00:08] Starting Stage 2: Classification Finetuning
[2025-11-20 22:00:08] Using device: cuda
Loading ESM-C model...
ESM-C model loaded.
[2025-11-20 22:00:21] Loading Stage 1 weights from output/checkpoints/stage1_final.pth
[2025-11-20 22:00:25] Stage 1 encoder weights loaded.
  Epoch 1, Step 10/50, Loss: 0.6954, Batch Acc: 65.00%
  Epoch 1, Step 20/50, Loss: 0.5432, Batch Acc: 75.00%
--- Epoch 1 Complete. Avg Train Loss: 0.6123, Avg Train Acc: 72.50% ---
Test Accuracy: 85.00%
```

**生成文件**:
- `output/logs/stage2_log.txt` - 训练日志
- `output/checkpoints/stage2_final.pth` - 最终模型
- `output/visualizations/stage2_epoch_*.png` - 可视化图片

#### 3. 预测

```bash
python3 predict.py path/to/your/sequences.fasta
```

**输出示例**:
```
[Predict] Using device: cuda
[Predict] Loading fine-tuned weights from output/checkpoints/stage2_final.pth...
[Predict] Loading sequences from sequences.fasta...
[Predict] Running inference...
[Predict] Progress: 100/100 (100.00%)

--- Prediction Results ---
Total sequences predicted: 100

> seq_001
  Predicted Class: 1
  Probabilities:   [Class 0: 0.2341, Class 1: 0.7659]

[Predict] Class 1 count: 65

[Predict] Full results saved to output/predictions_output.csv
[Predict] Class 1 CSV (with sequences) saved to output/predictions_output_class1.csv
[Predict] Class 1 FASTA saved to output/predictions_output_class1.fasta
[Predict] Generating embedding visualization...
Visualization saved to output/visualizations/prediction_vis.png
```

**生成文件**:
- `output/predictions_output.csv` - 所有序列的预测结果
- `output/predictions_output_class1.csv` - 预测为类别 1 的序列(含序列信息)
- `output/predictions_output_class1.fasta` - 预测为类别 1 的序列 FASTA 文件
- `output/visualizations/prediction_vis.png` - 预测结果可视化

## 配置参数

所有可调参数集中在 `config.py` 文件中:

### 模型参数

```python
ESMC_MODEL_NAME = "esmc_600m"  # 模型名称
EMBEDDING_DIM = 1152           # 嵌入维度 (ESM-C 600M)
```

### 阶段 I 参数

```python
STAGE1_EPOCHS = 20              # 训练轮数
STAGE1_BATCH_SIZE = 32          # 批次大小
STAGE1_LEARNING_RATE = 1e-4     # 学习率
SAVE_CHECKPOINT_INTERVAL = 5    # 检查点保存间隔 (epoch)
RESUME_FROM_CHECKPOINT = None   # 断点恢复路径
```

### 阶段 II 参数

```python
STAGE2_EPOCHS = 50              # 训练轮数
STAGE2_BATCH_SIZE = 32          # 批次大小
STAGE2_LEARNING_RATE = 1e-4     # 学习率
FREEZE_BASE_MODEL = True        # 是否冻结编码器
UNFREEZE_LAST_N_LAYERS = 2      # 解冻最后 N 层 (0=全部冻结)
```

### 评估与可视化参数

```python
EVAL_EPOCH_INTERVAL = 1                # 评估间隔 (epoch)
VISUALIZATION_EPOCH_INTERVAL = 10      # 可视化间隔 (epoch)
VISUALIZATION_METHOD = "UMAP"          # 降维方法: "UMAP" 或 "t-SNE"
```

### 路径参数

```python
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")           # 数据目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")        # 输出目录
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")               # 日志目录
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints") # 检查点目录
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations") # 可视化目录
```

## 输出说明

### 日志文件

所有日志文件保存在 `output/logs/`:

- **stage1_log.txt**: 阶段 I 训练日志,包含每个 epoch 的 Loss
- **stage2_log.txt**: 阶段 II 训练日志,包含 Loss、准确率等指标
- **predict_log.txt**: 预测日志,记录每个序列的预测结果

### 检查点文件

模型检查点保存在 `output/checkpoints/`:

- **stage1_epoch_N.pth**: 阶段 I 第 N 轮的检查点
- **stage1_final.pth**: 阶段 I 最终模型
- **stage2_final.pth**: 阶段 II 最终模型 (用于预测)

### 预测结果文件

#### predictions_output.csv

所有序列的预测结果:

```csv
sequence_name,predicted_class,prob_class_0,prob_class_1
seq_001,1,0.234567,0.765433
seq_002,0,0.891234,0.108766
```

#### predictions_output_class1.csv

预测为类别 1 的序列(含完整序列信息):

```csv
prob_class_1,sequence,predicted_class,sequence_name,prob_class_0
0.765433,MKTVRQERLK...,1,seq_001,0.234567
```

#### predictions_output_class1.fasta

预测为类别 1 的序列 FASTA 文件:

```fasta
>seq_001
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
```

### 可视化图片

降维可视化图片保存在 `output/visualizations/`:

- **stage1_epoch_N.png**: 阶段 I 第 N 轮的语义空间可视化
- **stage2_epoch_N.png**: 阶段 II 第 N 轮的语义空间可视化
- **prediction_vis.png**: 预测结果的语义空间分布

## 常见问题

### Q1: 如何调整训练轮数?

修改 `config.py` 中的 `STAGE1_EPOCHS` 和 `STAGE2_EPOCHS` 参数。

### Q2: 如何从断点恢复训练?

设置 `config.py` 中的 `RESUME_FROM_CHECKPOINT` 参数:

```python
RESUME_FROM_CHECKPOINT = "output/checkpoints/stage1_epoch_10.pth"
```

### Q3: 显存不足怎么办?

减小批次大小:

```python
STAGE1_BATCH_SIZE = 16  # 或更小
STAGE2_BATCH_SIZE = 16
```

### Q4: 如何调整编码器冻结策略?

修改 `config.py`:

```python
# 完全冻结编码器
FREEZE_BASE_MODEL = True
UNFREEZE_LAST_N_LAYERS = 0

# 解冻最后 2 层
FREEZE_BASE_MODEL = True
UNFREEZE_LAST_N_LAYERS = 2

# 完全不冻结
FREEZE_BASE_MODEL = False
```

### Q5: 可视化失败怎么办?

确保安装了 `umap-learn`:

```bash
python3 -m pip install umap-learn
```

如果数据集太小,可能会出现警告,但不影响训练和预测功能。

### Q6: 如何更改降维方法?

修改 `config.py`:

```python
VISUALIZATION_METHOD = "t-SNE"  # 或 "UMAP"
```

### Q7: 预测输出的类别含义是什么?

- **Class 0**: 对应 `train_mono.fasta` 中的类别 (例如:单铜蛋白)
- **Class 1**: 对应 `train_di.fasta` 中的类别 (例如:双铜蛋白)

## 技术支持

如有问题,请检查:

1. **日志文件**: 查看 `output/logs/` 中的日志文件
2. **环境配置**: 确保在 `esm3` 环境中运行
3. **数据格式**: 确保 FASTA 文件格式正确
4. **路径设置**: 确保 `config.py` 中的路径配置正确

---

**版本**: 1.0  
**最后更新**: 2025-11-20  
**作者**: Terry Wang
