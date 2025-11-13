import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# --- 路径修复 ---
# 将项目根目录 (scripts 文件夹的上一级) 添加到 sys.path
# 以便 model.py 中的 'from esm...' 导入可以找到 'esm' 包
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ----------------

# 'from model' 现在可以工作，因为它在 scripts 目录中
# 'model.py' 中的 'from esm' 也可以工作，因为 PROJECT_ROOT 在 sys.path 中
from model import get_model_and_tokenizer 
import os

# --- 配置参数 ---
MODEL_NAME = "esmc_600m"
# ! 你的本地权重路径 (绝对路径，保持不变)
LOCAL_WEIGHTS_PATH = "/home/wangty/esm/esm/data/weights/esmc_600m_2024_12_v0.pth" 
NUM_CLASSES = 2
UNFREEZE_LAYERS = 2
LEARNING_RATE = 1e-5
EPOCHS = 3 # 仅用于演示
BATCH_SIZE = 2
# ! 更新：将权重保存到项目根目录下的 'weights' 文件夹
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "weights", "esmc_classifier.pth")

# -----------------

def collate_fn(batch, pad_id):
    """
    自定义数据处理函数，用于动态 padding。
    """
    sequences, labels = zip(*batch)
    
    token_list = [
        torch.tensor(item) for item in sequences
    ]
    
    padded_tokens = torch.nn.utils.rnn.pad_sequence(
        token_list, 
        batch_first=True, 
        padding_value=pad_id
    )
    
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_tokens, labels

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # 1. 加载模型和 Tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name=MODEL_NAME,
        local_weights_path=LOCAL_WEIGHTS_PATH, # ! 传递本地路径
        num_classes=NUM_CLASSES,
        unfreeze_layers=UNFREEZE_LAYERS,
        device=device
    )
    model.train() # 设置为训练模式
    
    pad_id = tokenizer.pad_token_id

    # 2. 准备示例数据和 DataLoader
    # (这里使用固定的模拟数据，实际中应从文件加载)
    print("[Train] Preparing dummy data...")
    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPTS", # 类别 0
        "MKVLWAALLVTFLAGCQAKVEQ",                # 类别 1
        "MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNW",    # 类别 0
        "MGKVKVKKKGRKTGKSK"                      # 类别 1
    ]
    labels = [0, 1, 0, 1]
    
    # Tokenize
    tokenized_sequences = [
        tokenizer.encode(s, add_special_tokens=True) for s in sequences
    ]
    
    dataset = list(zip(tokenized_sequences, labels))
    
    # 使用 lambda 传入 pad_id
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, pad_id)
    )

    # 3. 设置优化器和损失函数
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    
    print(f"[Train] Optimizer configured to train {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")

    # 4. 训练循环
    print("[Train] Starting training loop...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_tokens, batch_labels in data_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(batch_tokens)
            
            # 计算损失
            loss = criterion(logits, batch_labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    # 5. 保存模型
    # ! 确保 weights 目录存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 只保存模型的状态字典 (推荐)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[Train] Model training complete. Weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()