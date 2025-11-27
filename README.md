# ESM-C 蛋白质分类工具集

基于 ESM-C (600M) 的蛋白质序列分类工具集，包含两种不同的训练策略。

## 📋 项目概述

本项目提供了两种蛋白质序列分类方案，针对不同的应用场景：

1. **`discovery/`** - 大规模数据挖掘（从海量序列中快速筛选目标蛋白）
2. **`classify/`** - 精细分类挖掘（小样本高质量分类与语义空间分析）

## 🗂️ 项目结构

```
esmc-wayawayawa/
├── discovery/         # 大规模数据挖掘方案
│   ├── train.py
│   ├── predict.py
│   └── README.md
├── classify/          # 精细分类挖掘方案（SETFIT 策略）
│   ├── stage1_pretrain.py
│   ├── stage2_finetune.py
│   ├── predict.py
│   └── README.md
├── datasets/          # 共享数据集目录
├── data/              # ESM-C 模型权重
│   └── weights/
├── sdk/               # Forge API 客户端
├── models/            # ESM 模型定义
├── tokenization/      # 分词器
└── README.md          # 本文档
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/Terry-Wang-Lynx/esmc-wayawayawa.git
cd esmc-wayawayawa

# 创建并激活环境
conda create -n esm3 python=3.10
conda activate esm3

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型权重

从 [Hugging Face](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/tree/main/data/weights) 下载 ESM-C 权重：

```bash
mkdir -p data/weights
# 下载 esmc_600m_2024_12_v0.pth 到 data/weights/
```

### 3. 选择方案

#### 方案 A：大规模数据挖掘（`discovery/`）

**适用场景**：
- 🔍 从 UniRef、SwissProt 等大型数据库中筛选目标蛋白
- 📊 处理极度不平衡数据（正负样本比例 1:100 甚至 1:1000）
- ⚡ 需要快速训练和高通量预测
- 💾 中等到大规模样本量（1000+ 条正样本）

**典型应用**：
- 从百万级序列库中挖掘特定功能蛋白（如 TIM-barrel、酪氨酸酶）
- 大规模蛋白质功能注释
- 高通量虚拟筛选

**使用方法**：
```bash
cd discovery
python3 train.py  # 训练
python3 predict.py  # 预测
```

详见 [`discovery/README.md`](discovery/README.md)

#### 方案 B：精细分类挖掘（`classify/`）

**适用场景**：
- 🎯 小样本高质量分类（< 100 条正样本）
- 📈 需要深入理解语义空间分布
- 🔬 精细区分高度相似的蛋白亚类
- 🧪 研究蛋白质序列-功能关系

**典型应用**：
- 区分单铜/双铜蛋白等精细亚型
- 蛋白质家族内部分类
- 探索性数据分析与可视化

**使用方法**：
```bash
cd classify
python3 stage1_pretrain.py  # 对比学习预训练
python3 stage2_finetune.py  # 分类头训练
python3 predict.py input.fasta  # 预测
```

详见 [`classify/README.md`](classify/README.md)

## 📊 方案对比

| 特性 | `discovery/` | `classify/` |
|------|--------------|-------------|
| **核心目标** | 大规模数据挖掘 | 精细分类挖掘 |
| **典型场景** | 从百万级库中筛选 | 小样本精准分类 |
| **训练策略** | 单阶段（直接分类） | 两阶段（对比学习 + 分类） |
| **适用数据量** | 中大规模（1000+） | 小样本（< 100） |
| **数据不平衡** | 极度不平衡（1:1000） | 相对平衡（1:1 到 1:10） |
| **训练时间** | 较短（单阶段） | 较长（两阶段） |
| **预测速度** | 快速（高通量） | 中等 |
| **可视化** | 基础（训练曲线） | 丰富（UMAP/t-SNE 语义空间） |
| **复杂度** | 低 | 高 |
| **泛化能力** | 中等 | 高 |

## 🛠️ 核心功能

### 共享特性

- ✅ **ESM-C 600M 基座**：使用 EvolutionaryScale 的蛋白质语言模型
- ✅ **灵活的层冻结策略**：可选择冻结/解冻编码器层
- ✅ **实时训练监控**：详细的日志和进度输出
- ✅ **多格式输出**：CSV、FASTA 等格式的预测结果

### `discovery/` 独有（大规模挖掘）

- 🔍 **高通量预测**：快速处理百万级序列
- ⚡ **动态负采样**：自动平衡极度不平衡数据
- 📈 **实时评估**：每轮自动在验证集上评估
- 💾 **周期性保存**：定期保存模型检查点，防止意外中断
- 📊 **训练曲线可视化**：Loss、Accuracy、F1 等指标实时绘图

### `classify/` 独有（精细分类）

- 🎯 **对比学习预训练**：通过 CosineEmbeddingLoss 优化语义空间
- 📊 **语义空间可视化**：UMAP/t-SNE 降维可视化，直观展示分类边界
- 🔄 **断点恢复**：支持训练中断后继续
- 🧪 **小样本优化**：SETFIT 策略，适合少量标注数据

## 📖 详细文档

- [对比学习方案详细文档](classify/README.md)
- [直接微调方案详细文档](discovery/README.md)

## 🔧 常见问题

### Q: 我应该选择哪个方案？

**选择 `discovery/`（大规模挖掘）如果：**
- 从 UniRef、SwissProt 等大型数据库中筛选目标蛋白
- 数据极度不平衡（正负样本比例 1:100+）
- 有 1000+ 条正样本
- 需要高通量预测

**选择 `classify/`（精细分类）如果：**
- 小样本场景（< 100 条正样本）
- 需要深入理解语义空间分布
- 精细区分高度相似的蛋白亚类
- 需要高质量可视化分析

### Q: 如何处理显存不足？

减小批次大小（`BATCH_SIZE`）或减少解冻层数（`UNFREEZE_LAYERS`）。

### Q: 训练需要多久？

- `classify/`：约 1-2 小时（取决于数据量和轮数）
- `discovery/`：约 30 分钟 - 1 小时

## 📝 引用

如果使用本项目，请引用：

```bibtex
@software{esmc_wayawayawa,
  author = {Terry Wang},
  title = {ESM-C Protein Classification Toolkit},
  year = {2025},
  url = {https://github.com/Terry-Wang-Lynx/esmc-wayawayawa}
}
```

## 📄 许可证

本项目基于 MIT 许可证开源。

---

**版本**: 2.0  
**最后更新**: 2025-11-27  
**作者**: Terry Wang
