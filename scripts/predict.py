import torch
import torch.nn.functional as F
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

from model import get_model_and_tokenizer # 从 scripts/model.py 导入
import os

# --- 配置参数 ---
MODEL_NAME = "esmc_600m"
# ! 你的本地权重路径 (绝对路径，保持不变)
LOCAL_WEIGHTS_PATH = "/home/wangty/esm/esm/data/weights/esmc_600m_2024_12_v0.pth"
NUM_CLASSES = 2
# ! 更新：从项目根目录下的 'weights' 文件夹加载
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "esmc_classifier.pth")
UNFREEZE_LAYERS = 0 # 预测时不需要解冻

# -----------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Using device: {device}")

    # 1. 加载模型结构和 Tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name=MODEL_NAME,
        local_weights_path=LOCAL_WEIGHTS_PATH, # ! 传递本地路径
        num_classes=NUM_CLASSES,
        unfreeze_layers=UNFREEZE_LAYERS,
        device=device
    )
    
    # 2. 加载我们微调过的权重
    if not os.path.exists(MODEL_PATH):
        print(f"[Predict] Error: Model weights not found at {MODEL_PATH}")
        print("Please run train.py first to generate the model weights.")
        return
        
    print(f"[Predict] Loading fine-tuned weights from {MODEL_PATH}...")
    # 加载状态字典
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("[Predict] Weights loaded successfully.")
    
    # 切换到评估模式
    model.eval()

    # 3. 准备待预测数据
    sequences_to_predict = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPTS", # 应该接近 0
        "MKVLWAALLVTFLAGCQAKVEQ",                # 应该接近 1
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDH" # 随机长序列
    ]
    
    pad_id = tokenizer.pad_token_id
    
    print(f"\n[Predict] Tokenizing {len(sequences_to_predict)} sequences for inference...")
    
    token_list = [
        torch.tensor(tokenizer.encode(s, add_special_tokens=True)) 
        for s in sequences_to_predict
    ]
    
    padded_tokens = torch.nn.utils.rnn.pad_sequence(
        token_list, 
        batch_first=True, 
        padding_value=pad_id
    ).to(device)

    # 4. 执行预测
    print("[Predict] Running inference...")
    with torch.no_grad(): # 关闭梯度计算
        logits = model(padded_tokens)
        
    # 将 logits 转换为概率
    probabilities = F.softmax(logits, dim=1)
    
    # 获取预测类别
    predictions = torch.argmax(probabilities, dim=1)

    # 5. 显示结果
    print("\n--- Prediction Results ---")
    for i, seq in enumerate(sequences_to_predict):
        print(f"\nSequence: {seq[:40]}...")
        print(f"  Logits (Raw):       [Class 0: {logits[i, 0]:.4f}, Class 1: {logits[i, 1]:.4f}]")
        print(f"  Probabilities:    [Class 0: {probabilities[i, 0]:.4f}, Class 1: {probabilities[i, 1]:.4f}]")
        print(f"  Predicted Class:  {predictions[i].item()}")

if __name__ == "__main__":
    main()