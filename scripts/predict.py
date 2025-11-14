import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys

# --- 路径修复 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ----------------

from model import get_model_and_tokenizer
# 导入 FastaDataset (用于读取) 和 PredictCollate (用于打包)
from data_loader import FastaDataset, PredictCollate

# --- 配置参数 ---
MODEL_NAME = "esmc_600m"
# ! 你的本地权重路径 (请修改)
LOCAL_WEIGHTS_PATH = "/home/wangty/esm/esm/data/weights/esmc_600m_2024_12_v0.pth"
NUM_CLASSES = 2

# ! 加载我们微调过的权重
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "best", "esmc_classifier_best.pth")

# ! 需要预测的 FASTA 文件
FASTA_TO_PREDICT = os.path.join(PROJECT_ROOT, "datasets", "train_positive.fasta")
# 预测时的批次大小
BATCH_SIZE = 8 
# -----------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Using device: {device}")

    # 1. 加载模型结构和 Tokenizer
    # 注意：unfreeze_layers=0，因为我们只是推理
    model, tokenizer = get_model_and_tokenizer(
        model_name=MODEL_NAME,
        local_weights_path=LOCAL_WEIGHTS_PATH,
        num_classes=NUM_CLASSES,
        unfreeze_layers=0, # 推理时不需要解冻
        device=device
    )
    
    # 2. 加载我们微调过的权重
    if not os.path.exists(MODEL_PATH):
        print(f"[Predict] Error: Model weights not found at {MODEL_PATH}")
        print("          Please run train.py first to generate weights.")
        return
        
    print(f"[Predict] Loading fine-tuned weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"[Predict] Error loading weights: {e}")
        print("          Ensure NUM_CLASSES matches the saved model.")
        return
        
    model.eval() # 切换到评估模式

    # 3. 准备待预测数据
    print(f"[Predict] Loading sequences from {FASTA_TO_PREDICT}...")
    if not os.path.exists(FASTA_TO_PREDICT):
        print(f"[Predict] Error: Input FASTA file not found.")
        return

    # 使用 FastaDataset (label=None)
    predict_dataset = FastaDataset(FASTA_TO_PREDICT, label=None)
    
    # 使用预测 Collate
    collate_fn = PredictCollate(tokenizer)
    
    data_loader = DataLoader(
        predict_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 预测时不需要打乱
        collate_fn=collate_fn,
        num_workers=0 # 简单任务用 0
    )

    # 4. 执行预测
    print("[Predict] Running inference...")
    results = []
    
    with torch.no_grad(): # 关闭梯度计算
        for names_batch, batch_tokens in data_loader:
            batch_tokens = batch_tokens.to(device)
            
            logits = model(batch_tokens)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # 收集结果
            for i in range(len(names_batch)):
                results.append({
                    "name": names_batch[i],
                    "prob_class_0": probabilities[i, 0].item(),
                    "prob_class_1": probabilities[i, 1].item(),
                    "predicted_class": predictions[i].item()
                })

    # 5. 显示结果
    print("\n--- Prediction Results ---")
    print(f"Total sequences predicted: {len(results)}")
    
    # (可选) 打印前 20 个结果
    for res in results[:20]:
        print(f"\n> {res['name']}")
        print(f"  Predicted Class: {res['predicted_class']}")
        print(f"  Probabilities:   [Class 0: {res['prob_class_0']:.4f}, Class 1: {res['prob_class_1']:.4f}]")
        
    # (可选) 将结果保存到文件
    output_file = os.path.join(PROJECT_ROOT, "predictions_output.csv")
    try:
        with open(output_file, 'w') as f:
            f.write("sequence_name,predicted_class,prob_class_0,prob_class_1\n")
            for res in results:
                f.write(f"{res['name']},{res['predicted_class']},{res['prob_class_0']:.6f},{res['prob_class_1']:.6f}\n")
        print(f"\n[Predict] Full results saved to {output_file}")
    except Exception as e:
        print(f"\n[Predict] Error saving results to file: {e}")


if __name__ == "__main__":
    main()