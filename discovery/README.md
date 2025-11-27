# Discovery - ESM-C 大规模数据挖掘方案

基于 ESM-C (600M) 的高通量蛋白质序列挖掘工具，专为从海量数据库中快速筛选目标蛋白而设计。

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [配置参数](#配置参数)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

## 项目简介

本工具专为**大规模蛋白质数据挖掘**设计，采用**单阶段直接微调**策略，能够从百万级序列库中高效筛选目标蛋白：

- **高通量预测**：快速处理 UniRef、SwissProt 等大型数据库
- **动态负采样**：自动平衡极度不平衡的正负样本（1:1000+）
- **实时监控**：每轮自动评估，及时发现过拟合
- **鲁棒训练**：周期性保存检查点，防止意外中断

### 典型应用场景

- 🔍 **功能蛋白挖掘**：从 UniRef50/90 中筛选特定功能蛋白（如 TIM-barrel、酪氨酸酶）
- 📊 **大规模注释**：对百万级未注释序列进行功能预测
- ⚡ **虚拟筛选**：高通量筛选候选药物靶点或工业酶
- 💾 **数据库构建**：构建特定功能的蛋白质子数据库

### 数据要求

- ✅ 中等到大规模正样本（1000+ 条）
- ✅ 海量负样本（可达百万级）
- ✅ 极度不平衡数据（正负比例 1:100 到 1:1000）
- ✅ 关注召回率和预测速度

## 核心特性

- ⚡ **动态负采样**：每轮从大量负样本中随机抽取，自动平衡训练
- 📈 **实时监控**：每轮输出训练/验证指标（Loss、Acc、Precision、Recall、F1）
- 💾 **周期性保存**：定期保存模型检查点，防止意外中断
- 📊 **自动可视化**：实时生成训练曲线图（Loss、Accuracy、F1 等）
- 🎯 **最佳模型保存**：自动保存验证集上表现最好的模型

## 项目结构

```
discovery/
├── train.py           # 🚀 训练脚本
├── predict.py         # 🔮 预测脚本
├── model.py           # 🧠 模型定义（ESM-C + 分类头）
├── data_loader.py     # 💾 数据加载与动态采样
├── README.md          # 📖 本文档
├── datasets/          # 📁 数据目录
│   └── TIM-barrel/    # 示例数据集
│       ├── train_positive.fasta
│       ├── train_negative.fasta
│       ├── test_positive.fasta
│       └── test_negative.fasta
└── outputs/           # 📂 输出目录
    └── TIM-barrel/    # 按数据集组织
        ├── training_log.csv
        ├── training_loss.png
        ├── validation_accuracy.png
        └── weights/
            ├── best/
            │   └── esmc_classifier_best.pth
            └── parameter/
                └── esmc_classifier_epoch_*.pth
```

## 环境配置

### 服务器环境

```bash
# 1. SSH 连接到服务器
ssh wangty@10.19.147.98

# 2. 激活 esm3 环境
conda activate esm3

# 3. 进入项目目录
cd esm/esm/discovery
```

### 依赖包

```bash
# 安装所有依赖
pip install -r requirements.txt
```

核心依赖包括：
- `torch` - PyTorch 深度学习框架
- `numpy` - 数值计算
- `matplotlib` - 可视化
- `scikit-learn` - 机器学习工具
- `esm` - ESM 蛋白质语言模型（项目已包含）

## 数据准备

### 数据格式

所有数据文件使用标准 FASTA 格式：

```fasta
>sequence_id_1
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>sequence_id_2
KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE
```

### 数据文件组织

为每个数据集创建独立目录：

```
datasets/
└── your_dataset/
    ├── train_positive.fasta   # 训练集 - 正样本
    ├── train_negative.fasta   # 训练集 - 负样本
    ├── test_positive.fasta    # 测试集 - 正样本
    └── test_negative.fasta    # 测试集 - 负样本
```

### 数据划分建议

1. **训练/测试比例**：80/20 或 70/30
2. **去冗余**：使用 CD-HIT 确保训练集和测试集之间序列相似度 < 70%
3. **负样本**：可以很大（几十万条），会自动动态采样

### 数据准备脚本

项目提供了 `prepare_data.py` 脚本用于：
- 自动划分正样本为训练/测试集
- 检测并删除负样本中与正样本重复的序列

```bash
# 编辑脚本中的路径
python3 prepare_data.py
```

## 使用方法

### 1. 配置训练参数

编辑 `train.py` 中的配置参数：

```python
# --- 配置参数 ---
MODEL_NAME = "esmc_600m"
LOCAL_WEIGHTS_PATH = "/path/to/esmc_600m_2024_12_v0.pth"
NUM_CLASSES = 2
UNFREEZE_LAYERS = 2
LEARNING_RATE = 1e-6
EPOCHS = 1000
EVAL_INTERVAL = 1  # 每多少个 epoch 评估一次
PARAMETER_SAVE_INTERVAL = 5  # 每多少个 epoch 保存一次
BATCH_SIZE = 16

# --- 训练集路径 ---
POSITIVE_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "train_positive.fasta")
NEGATIVE_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "train_negative.fasta")

# --- 测试集路径 ---
TEST_POSITIVE_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "test_positive.fasta")
TEST_NEGATIVE_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "test_negative.fasta")

# --- 动态采样 ---
USE_DYNAMIC_SAMPLING = True  # 推荐开启
```

### 2. 训练模型

```bash
# 前台运行（用于调试）
python3 train.py

# 后台运行（推荐）
nohup python3 train.py > outputs/TIM-barrel/train.log 2>&1 &

# 查看实时日志
tail -f outputs/TIM-barrel/train.log
```

**输出示例**：

```
[Train] Using device: cuda:3
[Model] Loading ESMC base model 'esmc_600m' from local path: /path/to/weights
[Model] Local weights loaded successfully.
[Model] Unfreezing last 2 transformer blocks...
[Train] Initializing Training dataset (Dynamic Sampling: True)...
[DataLoader] Loading sequences from: datasets/TIM-barrel/train_positive.fasta
[read_fasta] Loaded 2930 sequences from train_positive.fasta
[read_fasta] Loaded 558884 sequences from train_negative.fasta
[Train] Optimizer configured to train 31867778 parameters.
[Train] Total training steps per epoch: 367

  Epoch 1, Step 10/367, Loss: 0.6887, Batch Acc: 62.50%
  Epoch 1, Step 20/367, Loss: 0.6923, Batch Acc: 43.75%
  ...
--- Epoch 1 Complete. Avg Train Loss: 0.4917, Avg Train Acc: 87.94% ---
--- Epoch 1 Validation --- Loss: 0.2375, Acc: 99.77%, Precision: 99.86%, Recall: 99.59%, F1: 99.73%
    (New best model! Acc: 99.77%) [Saved to outputs/TIM-barrel/weights/best/esmc_classifier_best.pth]
```

### 3. 预测

编辑 `predict.py` 中的配置：

```python
# 模型路径
MODEL_PATH = os.path.join(SCRIPT_DIR, "outputs", "TIM-barrel", "weights", "best", "esmc_classifier_best.pth")

# 输入文件
FASTA_TO_PREDICT = os.path.join(PROJECT_ROOT, "datasets", "uniref50.fasta")

# 输出目录
output_dir = os.path.join(SCRIPT_DIR, "interference", "TIM-barrel")
```

运行预测：

```bash
python3 predict.py
```

**输出示例**：

```
[Predict] Using device: cuda:3
[Predict] Loading fine-tuned weights from outputs/TIM-barrel/weights/best/esmc_classifier_best.pth...
[Predict] Loading sequences from datasets/uniref50.fasta...
[Predict] Running inference...
[Predict] Progress: 1000/1000 (100.00%)

--- Prediction Results ---
Total sequences predicted: 1000

> seq_001
  Predicted Class: 1
  Probabilities:   [Class 0: 0.2341, Class 1: 0.7659]

[Predict] Class 1 count: 650
[Predict] Full results saved to interference/TIM-barrel/predictions_output.csv
[Predict] Class 1 CSV (with sequences) saved to interference/TIM-barrel/predictions_output_class1.csv
[Predict] Class 1 FASTA saved to interference/TIM-barrel/predictions_output_class1.fasta
```

## 配置参数

### 模型参数

```python
MODEL_NAME = "esmc_600m"           # 模型名称
LOCAL_WEIGHTS_PATH = "..."         # 本地权重路径
NUM_CLASSES = 2                    # 分类类别数
UNFREEZE_LAYERS = 2                # 解冻最后 N 层（0=全部冻结）
```

### 训练参数

```python
LEARNING_RATE = 1e-6               # 学习率（建议 1e-6 到 1e-5）
EPOCHS = 1000                      # 训练轮数
BATCH_SIZE = 16                    # 批次大小（根据显存调整）
USE_DYNAMIC_SAMPLING = True        # 是否使用动态负采样
```

### 评估与保存

```python
EVAL_INTERVAL = 1                  # 每多少轮评估一次
PARAMETER_SAVE_INTERVAL = 5        # 每多少轮保存一次检查点
```

### 数据路径

```python
POSITIVE_FASTA = "..."             # 训练正样本
NEGATIVE_FASTA = "..."             # 训练负样本
TEST_POSITIVE_FASTA = "..."        # 测试正样本
TEST_NEGATIVE_FASTA = "..."        # 测试负样本
```

## 输出说明

### 训练输出

#### 1. 日志文件（`training_log.csv`）

记录每轮的详细指标：

```csv
epoch,train_loss,train_accuracy,val_loss,val_accuracy,val_precision,val_recall,val_f1,best_val_accuracy,learning_rate
1,0.4917,0.8794,0.2375,0.9977,0.9986,0.9959,0.9973,0.9977,1e-06
2,0.2134,0.9523,0.1892,0.9983,0.9991,0.9968,0.9979,0.9983,1e-06
```

#### 2. 可视化图片

- **training_loss.png**：训练/验证 Loss 曲线
- **validation_accuracy.png**：训练/验证准确率曲线
- **val_prf1.png**：验证集 Precision/Recall/F1 曲线
- **learning_rate.png**：学习率变化曲线

#### 3. 模型检查点

- **weights/best/esmc_classifier_best.pth**：验证集上表现最好的模型
- **weights/parameter/esmc_classifier_epoch_N.pth**：每 N 轮保存的检查点

### 预测输出

#### 1. predictions_output.csv

所有序列的预测结果：

```csv
sequence_name,predicted_class,prob_class_0,prob_class_1
seq_001,1,0.234567,0.765433
seq_002,0,0.891234,0.108766
```

#### 2. predictions_output_class1.csv

预测为 Class 1 的序列（含完整序列）：

```csv
prob_class_1,sequence,predicted_class,sequence_name,prob_class_0
0.765433,MKTVRQERLK...,1,seq_001,0.234567
```

#### 3. predictions_output_class1.fasta

预测为 Class 1 的序列 FASTA 文件：

```fasta
>seq_001
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
```

## 常见问题

### Q1: 什么是动态负采样？

**答**：当负样本数量远大于正样本时（如 1:100），动态负采样会在每个 epoch：
- 使用全部正样本（如 3000 条）
- 从负样本池（如 300,000 条）中随机抽取等量负样本（3000 条）
- 这样每轮都是 1:1 平衡训练，且随着训练进行会覆盖大部分负样本

### Q2: 如何判断是否过拟合？

**答**：查看可视化图片：
- **训练 Loss 持续下降，验证 Loss 不再下降或上升** → 过拟合
- **训练/验证 Loss 同步下降** → 正常
- 建议在验证 Loss 不再下降后停止训练（Early Stopping）

### Q3: 显存不足怎么办？

**答**：
1. 减小 `BATCH_SIZE`（如从 16 改为 8）
2. 减少 `UNFREEZE_LAYERS`（如从 2 改为 1 或 0）
3. 使用更小的模型（如 ESM-C 300M）

### Q4: 训练速度太慢怎么办？

**答**：
1. 增大 `BATCH_SIZE`（如果显存允许）
2. 减少 `EVAL_INTERVAL`（如从 1 改为 5，减少评估频率）
3. 使用 Flash Attention 加速（需要 CUDA 11.6+）

### Q5: 如何调整学习率？

**答**：
- **过拟合严重**：降低学习率（如 1e-6）
- **收敛太慢**：提高学习率（如 1e-5）
- **Loss 震荡**：降低学习率或减小 batch size

### Q6: 如何选择解冻层数？

**答**：
- **数据量 < 1000**：`UNFREEZE_LAYERS = 0`（完全冻结）
- **数据量 1000-5000**：`UNFREEZE_LAYERS = 1-2`
- **数据量 > 5000**：`UNFREEZE_LAYERS = 2-4`

### Q7: 验证集准确率虚高怎么办？

**答**：可能是数据泄露，建议：
1. 使用 CD-HIT 去除训练/测试集之间的高相似度序列（< 70%）
2. 增加测试集规模
3. 检查是否有重复序列

## 技术支持

如有问题，请检查：

1. **日志文件**：查看 `outputs/*/train.log` 和 `training_log.csv`
2. **可视化图片**：查看 Loss 和 Accuracy 曲线
3. **环境配置**：确保在 `esm3` 环境中运行
4. **数据格式**：确保 FASTA 文件格式正确
5. **路径设置**：确保所有路径配置正确

---

**版本**: 1.0  
**最后更新**: 2025-11-27  
**作者**: Terry Wang
