import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 消除 tokenizers 并行警告
import sys
import csv
import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，方便在服务器上保存图像
import matplotlib.pyplot as plt
import numpy as np

# --- 路径修复 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ----------------

from model import get_model_and_tokenizer 
from data_loader import TrainingDataset, TrainCollate # 导入不变

# --- 配置参数 ---
MODEL_NAME = "esmc_600m"
LOCAL_WEIGHTS_PATH = "/home/wangty/esm/esm/data/weights/esmc_600m_2024_12_v0.pth" 
NUM_CLASSES = 2
UNFREEZE_LAYERS = 4
LEARNING_RATE = 1e-5
EPOCHS = 1000
EVAL_INTERVAL = 10  # 每多少个 epoch 在测试集上评估一次 (与可视化频率保持一致)
PARAMETER_SAVE_INTERVAL = 25  # 每多少个 epoch 保存一次完整模型参数
VISUALIZATION_INTERVAL = 10  # 每多少个 epoch 生成一次降维可视化
BATCH_SIZE = 16 # 根据显存调整

# --- 训练集路径 ---
# --- 训练集路径 ---
POSITIVE_FASTA = "/home/wangty/esm/esm/discovery/datasets/l-dopa/train_positive.fasta"
NEGATIVE_FASTA = "/home/wangty/esm/esm/discovery/datasets/l-dopa/train_negative.fasta"

# --- 新增：测试集 (验证集) 路径 ---
TEST_POSITIVE_FASTA = "/home/wangty/esm/esm/discovery/datasets/l-dopa/test_positive.fasta"
TEST_NEGATIVE_FASTA = "/home/wangty/esm/esm/discovery/datasets/l-dopa/test_negative.fasta"

# --- 修改：推荐开启动态采样以解决不平衡问题 ---
USE_DYNAMIC_SAMPLING = True # 设为 True

MODEL_SAVE_PATH = "/home/wangty/esm/esm/discovery/outputs/III-copper/weights/esmc_classifier.pth"

WEIGHTS_ROOT = "/home/wangty/esm/esm/discovery/outputs/III-copper/weights"
BEST_MODEL_DIR = os.path.join(WEIGHTS_ROOT, "best")
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "esmc_classifier_best.pth")
PARAMETER_DIR = os.path.join(WEIGHTS_ROOT, "parameter")

# outputs 目录：日志和可视化图像
OUTPUTS_ROOT = "/home/wangty/esm/esm/discovery/outputs/III-copper"
LOG_FILE_PATH = os.path.join(OUTPUTS_ROOT, "training_log.csv")
PLOT_LOSS_PATH = os.path.join(OUTPUTS_ROOT, "training_loss.png")
PLOT_ACC_PATH = os.path.join(OUTPUTS_ROOT, "validation_accuracy.png")

# 可视化配置
VISUALIZATION_METHOD = 'UMAP'  # 'UMAP' or 't-SNE'
VISUALIZATION_DIR = os.path.join(OUTPUTS_ROOT, "visualizations")

# -----------------


# --- 新增：评估函数 ---
def evaluate_model(model, data_loader, device, criterion=None):
    """
    在给定的数据集上评估模型指标（损失、准确率、精确率、召回率、F1）
    默认将类别 1 视为正类（适用于二分类）
    """
    model.eval()  # 切换到评估模式 (关闭 dropout 等)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 用于二分类统计
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():  # 在评估时关闭梯度计算
        for batch_tokens, batch_labels, batch_names in data_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)

            # 前向传播
            logits = model(batch_tokens)

            # 损失
            if criterion is not None:
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()

            # 预测 (选择概率最高的类别)
            predictions = torch.argmax(logits, dim=1)

            # 准确率统计
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            # 将标签和预测展开用于二分类统计，约定类别 1 为正类
            pos_label = 1
            tp = ((predictions == pos_label) & (batch_labels == pos_label)).sum().item()
            fp = ((predictions == pos_label) & (batch_labels != pos_label)).sum().item()
            fn = ((predictions != pos_label) & (batch_labels == pos_label)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

    model.train()  # 将模型切回训练模式

    # 计算指标
    if total_samples > 0:
        accuracy = total_correct / total_samples
    else:
        accuracy = 0.0

    if criterion is not None and len(data_loader) > 0:
        avg_loss = total_loss / len(data_loader)
    else:
        avg_loss = 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


def update_log_file(log_path, epoch, train_loss, train_accuracy,
                    val_loss, val_accuracy, val_precision, val_recall, val_f1,
                    best_val_accuracy, learning_rate):
    """
    将每一轮的训练结果追加写入到 CSV 日志文件中
    """
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        # 如果文件不存在，先写入表头
        if not file_exists:
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "best_val_accuracy",
                "learning_rate",
            ])
        writer.writerow([
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            best_val_accuracy,
            learning_rate,
        ])




def generate_embedding_visualization(model, train_loader, test_loader, device, output_dir, method='UMAP', epoch=None):
    """
    生成训练集和测试集的降维可视化图
    
    Args:
        model: 训练好的模型
        train_loader: 训练集 DataLoader
        test_loader: 测试集 DataLoader
        device: 设备
        output_dir: 输出目录
        method: 降维方法 ('UMAP' or 't-SNE')
        epoch: 当前 epoch 编号（用于文件命名）
    """
    print(f"\n[Visualization] Generating embedding visualization using {method}...")
    
    model.eval()
    
    # 提取训练集 embeddings
    train_embeddings = []
    train_labels = []
    train_names = []
    with torch.no_grad():
        for batch_tokens, batch_labels, batch_names in train_loader:
            batch_tokens = batch_tokens.to(device)
            if hasattr(model, 'forward_encoder'):
                emb = model.forward_encoder(batch_tokens)
            else:
                raise AttributeError("Model does not have 'forward_encoder' method. Please check model.py.")
            train_embeddings.append(emb.cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())
            train_names.extend(batch_names)
    
    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_labels = np.array(train_labels)
    
    # 提取测试集 embeddings
    test_embeddings = []
    test_labels = []
    test_names = []
    with torch.no_grad():
        for batch_tokens, batch_labels, batch_names in test_loader:
            batch_tokens = batch_tokens.to(device)
            if hasattr(model, 'forward_encoder'):
                emb = model.forward_encoder(batch_tokens)
            else:
                raise AttributeError("Model does not have 'forward_encoder' method. Please check model.py.")
            test_embeddings.append(emb.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
            test_names.extend(batch_names)
    
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.array(test_labels)
    
    print(f"[Visualization] Train embeddings shape: {train_embeddings.shape}")
    print(f"[Visualization] Test embeddings shape: {test_embeddings.shape}")
    
    # 合并所有数据进行降维（确保在同一空间）
    all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    
    # 降维
    print(f"[Visualization] Performing dimensionality reduction...")
    try:
        if method == 'UMAP':
            try:
                import umap
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
            except ImportError:
                print("[Warning] UMAP not installed, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto')
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto')
        
        all_embeddings_2d = reducer.fit_transform(all_embeddings)
    except Exception as e:
        print(f"[Error] Dimensionality reduction failed: {e}")
        return
    
    # 拆分回训练集和测试集
    n_train = len(train_embeddings)
    train_2d = all_embeddings_2d[:n_train]
    test_2d = all_embeddings_2d[n_train:]
    
    # 保存坐标到 CSV
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据是否有 epoch 参数决定文件名
    if epoch is not None:
        coords_file = os.path.join(output_dir, f"embedding_coordinates_epoch_{epoch}.csv")
        vis_path = os.path.join(output_dir, f"embedding_visualization_epoch_{epoch}.png")
    else:
        coords_file = os.path.join(output_dir, "embedding_coordinates.csv")
        vis_path = os.path.join(output_dir, "embedding_visualization.png")
    
    print(f"[Visualization] Saving coordinates to {coords_file}...")
    
    with open(coords_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'label', 'label_name', 'name', 'x', 'y'])
        
        # 写入训练集数据
        for i in range(len(train_2d)):
            label_name = 'positive' if train_labels[i] == 1 else 'negative'
            writer.writerow(['train', train_labels[i], label_name, train_names[i], train_2d[i, 0], train_2d[i, 1]])
        
        # 写入测试集数据
        for i in range(len(test_2d)):
            label_name = 'positive' if test_labels[i] == 1 else 'negative'
            writer.writerow(['test', test_labels[i], label_name, test_names[i], test_2d[i, 0], test_2d[i, 1]])
    
    # 绘制可视化图
    print(f"[Visualization] Generating plot...")
    
    plt.figure(figsize=(14, 12))
    
    # 绘制训练集（浅色，小点）
    train_neg_mask = train_labels == 0
    train_pos_mask = train_labels == 1
    plt.scatter(train_2d[train_neg_mask, 0], train_2d[train_neg_mask, 1],
                c='lightblue', label='Train Negative', alpha=0.4, s=20, marker='o', edgecolors='blue', linewidths=0.3)
    plt.scatter(train_2d[train_pos_mask, 0], train_2d[train_pos_mask, 1],
                c='lightcoral', label='Train Positive', alpha=0.4, s=20, marker='o', edgecolors='red', linewidths=0.3)
    
    # 绘制测试集（深色，大点，不同标记）
    test_neg_mask = test_labels == 0
    test_pos_mask = test_labels == 1
    plt.scatter(test_2d[test_neg_mask, 0], test_2d[test_neg_mask, 1],
                c='blue', label='Test Negative', alpha=0.8, s=50, marker='X', edgecolors='darkblue', linewidths=0.8)
    plt.scatter(test_2d[test_pos_mask, 0], test_2d[test_pos_mask, 1],
                c='red', label='Test Positive', alpha=0.8, s=50, marker='X', edgecolors='darkred', linewidths=0.8)
    
    plt.title(f"Protein Embeddings Visualization ({method})\nTrain (Circle) vs Test (Cross)", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    
    print(f"[Visualization] Visualization saved to {vis_path}")
    print(f"[Visualization] Coordinates saved to {coords_file}")
    model.train()  # 切回训练模式


def save_training_plots(history, output_dir):
    """
    根据当前 history 绘制并保存训练曲线图像
    history: dict，包含
        'train_loss', 'train_accuracy',
        'val_loss', 'val_accuracy',
        'val_precision', 'val_recall', 'val_f1',
        'learning_rate',
        'val_epochs'  # 实际进行验证的 epoch 列表
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if not epochs:
        return

    # 1. 训练损失曲线
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    # 只绘制实际进行验证的 epoch 的验证损失
    if "val_epochs" in history and len(history["val_epochs"]) > 0:
        val_epochs = history["val_epochs"]
        # 直接根据 val_epochs 索引提取对应的值，不过滤 0 值
        val_losses = [history["val_loss"][e-1] for e in val_epochs]
        if len(val_losses) > 0:
            plt.plot(val_epochs, val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    # 2. 准确率曲线
    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    # 只绘制实际进行验证的 epoch 的验证准确率
    if "val_epochs" in history and len(history["val_epochs"]) > 0:
        val_epochs = history["val_epochs"]
        val_accs = [history["val_accuracy"][e-1] for e in val_epochs]
        if len(val_accs) > 0:
            plt.plot(val_epochs, val_accs, label="Val Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_accuracy.png"))
    plt.close()

    # 3. 验证集 Precision / Recall / F1 曲线
    if "val_epochs" in history and len(history["val_epochs"]) > 0:
        val_epochs = history["val_epochs"]
        val_precisions = [history["val_precision"][e-1] for e in val_epochs]
        val_recalls = [history["val_recall"][e-1] for e in val_epochs]
        val_f1s = [history["val_f1"][e-1] for e in val_epochs]
        
        plt.figure()
        if len(val_precisions) > 0:
            plt.plot(val_epochs, val_precisions, label="Val Precision", marker='o')
        if len(val_recalls) > 0:
            plt.plot(val_epochs, val_recalls, label="Val Recall", marker='s')
        if len(val_f1s) > 0:
            plt.plot(val_epochs, val_f1s, label="Val F1", marker='^')
            
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Precision / Recall / F1")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_prf1.png"))
        plt.close()

    # 4. 学习率曲线（如果有记录）
    if len(history.get("learning_rate", [])) == len(epochs):
        plt.figure()
        plt.plot(epochs, history["learning_rate"], label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_rate.png"))
        plt.close()


# --- 主函数 ---
def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # 1. 加载模型和 Tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name=MODEL_NAME,
        local_weights_path=LOCAL_WEIGHTS_PATH,
        num_classes=NUM_CLASSES,
        unfreeze_layers=UNFREEZE_LAYERS,
        device=device
    )
    
    # 2. 准备训练集
    print(f"[Train] Initializing Training dataset (Dynamic Sampling: {USE_DYNAMIC_SAMPLING})...")
    try:
        train_dataset = TrainingDataset(
            positive_path=POSITIVE_FASTA,
            negative_path=NEGATIVE_FASTA,
            dynamic_negative_sampling=USE_DYNAMIC_SAMPLING # 使用配置
        )
    except Exception as e:
        print(f"[Train] Error loading training dataset: {e}")
        return

    collate_fn = TrainCollate(tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )

    # --- 新增：准备测试集 (验证集) ---
    print("[Train] Initializing Test dataset...")
    try:
        # 测试集必须是固定的，所以 dynamic_negative_sampling 必须为 False
        test_dataset = TrainingDataset(
            positive_path=TEST_POSITIVE_FASTA,
            negative_path=TEST_NEGATIVE_FASTA,
            dynamic_negative_sampling=False # 测试集绝不能使用动态采样
        )
    except Exception as e:
        print(f"[Train] Error loading test dataset: {e}")
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2, # 评估时批次可以开大一点
        shuffle=False, # 评估时不需要打乱
        collate_fn=collate_fn, # 复用同一个 collate_fn
        num_workers=0
    )
    
    # 检查测试集是否为空
    if len(test_loader) == 0:
        print("[Warning] Test dataset is empty. Validation accuracy will not be calculated.")

    # 3. 设置优化器和损失函数
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    
    print(f"[Train] Optimizer configured to train {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")
    print(f"[Train] Total training steps per epoch: {len(train_loader)}")

    # Ensure directories for saving weights exist
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(PARAMETER_DIR, exist_ok=True)

    # Ensure directory for outputs (logs and plots) exists
    os.makedirs(OUTPUTS_ROOT, exist_ok=True)

    # 4. 训练循环
    print("[Train] Starting training loop...")
    best_val_accuracy = 0.0 # 用于保存最佳模型

    # 训练过程指标记录，用于日志和可视化
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "learning_rate": [],
        "best_val_accuracy": [],
        "val_epochs": []  # 记录实际进行验证的 epoch
    }

    # --- 新增：在训练开始前生成 Epoch 0 可视化 ---
    print("\n" + "="*80)
    print("[Train] Generating initial (Epoch 0) dimensionality reduction visualization...")
    print("="*80)
    try:
        generate_embedding_visualization(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            output_dir=VISUALIZATION_DIR,
            method=VISUALIZATION_METHOD,
            epoch=0  # 使用 0 表示初始状态
        )
        print("[Train] Initial visualization complete!")
    except Exception as e:
        print(f"[Train] Warning: Initial visualization generation failed: {e}")
        import traceback
        traceback.print_exc()

    for epoch in range(EPOCHS):
        model.train()  # 确保模型处于训练模式
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, (batch_tokens, batch_labels, batch_names) in enumerate(train_loader):
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            try:
                logits = model(batch_tokens)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                # 训练损失累计
                total_loss += loss.item()

                # 训练准确率累计
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

                if (i + 1) % 10 == 0:
                    batch_acc = (predictions == batch_labels).float().mean().item() if batch_labels.size(0) > 0 else 0.0
                    print(
                        f"  Epoch {epoch + 1}, Step {i + 1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, Batch Acc: {batch_acc * 100:.2f}%"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"--- [OOM Warning] Epoch {epoch + 1}, Step {i + 1}. Skipping batch. ---")
                    torch.cuda.empty_cache()
                else:
                    raise e

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(
            f"--- Epoch {epoch + 1} Complete. "
            f"Avg Train Loss: {avg_loss:.4f}, Avg Train Acc: {train_accuracy * 100:.2f}% ---"
        )

        # 当前学习率（考虑到可能使用不同的 scheduler 或 per-param-group lr，这里简单取第一个 param_group）
        current_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else LEARNING_RATE

        # --- 按配置间隔进行一次评估 ---
        val_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0

        # 仅在达到评估间隔或最后一轮时进行验证集评估
        if len(test_loader) > 0 and (((epoch + 1) % EVAL_INTERVAL == 0) or (epoch + 1 == EPOCHS)):
            # 记录当前 epoch 进行了验证
            history["val_epochs"].append(epoch + 1)
            
            val_metrics = evaluate_model(model, test_loader, device, criterion=criterion)
            val_loss = val_metrics["loss"]
            val_accuracy = val_metrics["accuracy"]
            val_precision = val_metrics["precision"]
            val_recall = val_metrics["recall"]
            val_f1 = val_metrics["f1"]

            print(
                f"--- Epoch {epoch + 1} Validation --- "
                f"Loss: {val_loss:.4f}, Acc: {val_accuracy * 100:.2f}%, "
                f"Precision: {val_precision * 100:.2f}%, Recall: {val_recall * 100:.2f}%, "
                f"F1: {val_f1 * 100:.2f}%"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # 实时保存当前最佳模型到 best 目录
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(
                    f"    (New best model! Acc: {best_val_accuracy * 100:.2f}%) "
                    f"[Saved to {BEST_MODEL_PATH}]"
                )

        # 每 PARAMETER_SAVE_INTERVAL 轮保存一次完整模型参数到 parameter 目录
        if (epoch + 1) % PARAMETER_SAVE_INTERVAL == 0:
            epoch_save_path = os.path.join(PARAMETER_DIR, f"esmc_classifier_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_save_path)
            print(f"    [Periodic Save] Epoch {epoch + 1} weights saved to {epoch_save_path}")

        # 更新本轮的指标记录（即使没有验证集，也记录训练相关的）
        history["train_loss"].append(avg_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)
        history["best_val_accuracy"].append(best_val_accuracy)

        # 每一轮循环结束后实时更新日志文件
        update_log_file(
            LOG_FILE_PATH,
            epoch + 1,
            avg_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            best_val_accuracy,
            current_lr,
        )

        # 每一轮循环结束后实时更新训练曲线图像
        save_training_plots(history, OUTPUTS_ROOT)
        
        # 每 VISUALIZATION_INTERVAL 轮生成一次降维可视化
        if (epoch + 1) % VISUALIZATION_INTERVAL == 0:
            print("\n" + "="*80)
            print(f"[Train] Generating dimensionality reduction visualization at epoch {epoch + 1}...")
            print("="*80)
            try:
                generate_embedding_visualization(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    output_dir=VISUALIZATION_DIR,
                    method=VISUALIZATION_METHOD,
                    epoch=epoch + 1  # 传入 epoch 编号用于文件命名
                )
                print(f"[Train] Visualization for epoch {epoch + 1} complete!")
            except Exception as e:
                print(f"[Train] Warning: Visualization generation failed: {e}")
                import traceback
                traceback.print_exc()



    # 5. 保存最终模型
    # (您也可以选择只保存在验证集上表现最好的模型)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[Train] Model training complete. Final weights saved to {MODEL_SAVE_PATH}")
    print(f"[Train] Best validation accuracy achieved: {best_val_accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()