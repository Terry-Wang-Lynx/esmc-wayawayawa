import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import csv
import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，方便在服务器上保存图像
import matplotlib.pyplot as plt

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
UNFREEZE_LAYERS = 2
LEARNING_RATE = 1e-5
EPOCHS = 1000
BATCH_SIZE = 4 # 根据显存调整

# --- 训练集路径 ---
POSITIVE_FASTA = os.path.join(PROJECT_ROOT, "datasets", "test.fasta")
NEGATIVE_FASTA = os.path.join(PROJECT_ROOT, "datasets", "train_negative.fasta")

# --- 新增：测试集 (验证集) 路径 ---
TEST_POSITIVE_FASTA = os.path.join(PROJECT_ROOT, "datasets", "test_positive.fasta")
TEST_NEGATIVE_FASTA = os.path.join(PROJECT_ROOT, "datasets", "test_negative.fasta")

# --- 修改：推荐开启动态采样以解决不平衡问题 ---
USE_DYNAMIC_SAMPLING = True # 设为 True

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "weights", "esmc_classifier.pth")

WEIGHTS_ROOT = os.path.join(PROJECT_ROOT, "weights")
BEST_MODEL_DIR = os.path.join(WEIGHTS_ROOT, "best")
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "esmc_classifier_best.pth")
PARAMETER_DIR = os.path.join(WEIGHTS_ROOT, "parameter")

# outputs 目录：日志和可视化图像
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
LOG_FILE_PATH = os.path.join(OUTPUTS_ROOT, "training_log.csv")
PLOT_LOSS_PATH = os.path.join(OUTPUTS_ROOT, "training_loss.png")
PLOT_ACC_PATH = os.path.join(OUTPUTS_ROOT, "validation_accuracy.png")
# -----------------


# --- 新增：评估函数 ---
def evaluate_model(model, data_loader, device):
    """
    在给定的数据集上评估模型准确率
    """
    model.eval() # 切换到评估模式 (关闭 dropout 等)
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad(): # 在评估时关闭梯度计算
        for batch_tokens, batch_labels in data_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            logits = model(batch_tokens)
            
            # 获取预测 (选择概率最高的类别)
            predictions = torch.argmax(logits, dim=1)
            
            # 统计正确数量
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
    model.train() # 将模型切回训练模式
    
    # 计算准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def update_log_file(log_path, epoch, train_loss, val_accuracy, best_val_accuracy):
    """
    将每一轮的训练结果追加写入到 CSV 日志文件中
    """
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        # 如果文件不存在，先写入表头
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_accuracy", "best_val_accuracy"])
        writer.writerow([epoch, train_loss, val_accuracy, best_val_accuracy])


def save_training_plots(history, output_dir):
    """
    根据当前 history 绘制并保存训练曲线图像
    history: dict，包含 'train_loss', 'val_accuracy'
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if not epochs:
        return

    # 训练损失曲线
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    # 验证准确率曲线（如果有）
    if len(history["val_accuracy"]) == len(epochs):
        plt.figure()
        plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_accuracy.png"))
        plt.close()


# --- 主函数 ---
def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
        "val_accuracy": [],
        "best_val_accuracy": []
    }

    for epoch in range(EPOCHS):
        model.train() # 确保模型处于训练模式
        total_loss = 0
        for i, (batch_tokens, batch_labels) in enumerate(train_loader):
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            try:
                logits = model(batch_tokens)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"--- [OOM Warning] Epoch {epoch+1}, Step {i+1}. Skipping batch. ---")
                    torch.cuda.empty_cache()
                else:
                    raise e
            
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"--- Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f} ---")

        # --- 修改：每 10 轮进行一次评估 ---
        if len(test_loader) > 0 and (epoch + 1) % 100 == 0:
            val_accuracy = evaluate_model(model, test_loader, device)
            print(f"--- Epoch {epoch+1} Validation Accuracy: {val_accuracy * 100:.2f}% ---")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # 实时保存当前最佳模型到 best 目录
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"    (New best model! Accuracy: {best_val_accuracy * 100:.2f}%) [Saved to {BEST_MODEL_PATH}]")
        
            # 每 25 轮保存一次完整模型参数到 parameter 目录
            if (epoch + 1) % 200 == 0:
                epoch_save_path = os.path.join(PARAMETER_DIR, f"esmc_classifier_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), epoch_save_path)
                print(f"    [Periodic Save] Epoch {epoch + 1} weights saved to {epoch_save_path}")

            # 更新本轮的指标记录
            history["train_loss"].append(avg_loss)
            history["val_accuracy"].append(val_accuracy)
            history["best_val_accuracy"].append(best_val_accuracy)

            # 每一轮循环结束后实时更新日志文件
            update_log_file(LOG_FILE_PATH, epoch + 1, avg_loss, val_accuracy, best_val_accuracy)

            # 每一轮循环结束后实时更新训练曲线图像
            save_training_plots(history, OUTPUTS_ROOT)

    # 5. 保存最终模型
    # (您也可以选择只保存在验证集上表现最好的模型)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[Train] Model training complete. Final weights saved to {MODEL_SAVE_PATH}")
    print(f"[Train] Best validation accuracy achieved: {best_val_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()