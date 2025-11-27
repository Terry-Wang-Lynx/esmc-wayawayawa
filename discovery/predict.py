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
MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "tyrosinase", "esmc_classifier_epoch_925.pth")

# ! 需要预测的 FASTA 文件
FASTA_TO_PREDICT = os.path.join(PROJECT_ROOT, "datasets", "uniref50.fasta")
# 预测时的批次大小
BATCH_SIZE = 32 
# 从中间开始预测的起始序列编号（从 0 开始计数），例如 5000 表示跳过前 5000 条序列
START_INDEX = 0
# -----------------

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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
    total_seqs = len(predict_dataset)
    processed_seqs = 0

    # 4. 执行预测并实时写入结果文件
    print("[Predict] Running inference...")
    results = []
    class1_results = []
    
    output_dir = os.path.join(SCRIPT_DIR, "interference", "III-copper")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "predictions_output.csv")
    class1_file = os.path.join(output_dir, "predictions_output_class1.csv")
    class1_fasta_file = os.path.join(output_dir, "predictions_output_class1.fasta")

    class1_f = open(class1_file, 'w')
    # 在 class1 的 CSV 中增加 sequence 列
    class1_f.write("prob_class_1,sequence,predicted_class,sequence_name,prob_class_0\n")

    class1_fasta_f = open(class1_fasta_file, 'w')

    try:
        with open(output_file, 'w') as f:
            # 写入主结果 CSV 的表头（保持不变）
            f.write("sequence_name,predicted_class,prob_class_0,prob_class_1\n")
    
            with torch.no_grad():  # 关闭梯度计算
                for names_batch, batch_tokens, seqs_batch in data_loader:
                    # 跳过前 START_INDEX 条序列，用于任务恢复
                    if processed_seqs + len(names_batch) <= START_INDEX:
                        processed_seqs += len(names_batch)
                        continue

                    batch_tokens = batch_tokens.to(device)
    
                    logits = model(batch_tokens)
                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
    
                    # 收集结果并实时写入文件
                    for i in range(len(names_batch)):
                        # 如果 START_INDEX 在 batch 内，跳过 batch 前面的部分
                        if processed_seqs + i < START_INDEX:
                            continue

                        name = names_batch[i]
                        seq = seqs_batch[i]  # 已在 PredictCollate 中清洗过（无空白）
    
                        res = {
                            "name": name,
                            "prob_class_0": probabilities[i, 0].item(),
                            "prob_class_1": probabilities[i, 1].item(),
                            "predicted_class": predictions[i].item(),
                            "sequence": seq
                        }
                        results.append(res)
    
                        if res['predicted_class'] == 1:
                            class1_results.append(res)
                            # class 1 的 CSV：增加 sequence 列，并保持实时写入
                            class1_f.write(
                                f"{res['prob_class_1']:.6f},"
                                f"{res['sequence']},"
                                f"{res['predicted_class']},"
                                f"{res['name']},"
                                f"{res['prob_class_0']:.6f}\n"
                            )
                            class1_f.flush()
                            os.fsync(class1_f.fileno())
    
                            # class 1 的 FASTA：构建二次筛选用的 FASTA 文件，同样实时写入
                            if res["sequence"]:
                                class1_fasta_f.write(
                                    f">{res['name']}\n{res['sequence']}\n"
                                )
                                class1_fasta_f.flush()
                                os.fsync(class1_fasta_f.fileno())
    
                        # 写入一行并立即刷新（主 CSV 与原逻辑保持一致）
                        f.write(
                            f"{res['name']},{res['predicted_class']},"
                            f"{res['prob_class_0']:.6f},{res['prob_class_1']:.6f}\n"
                        )
                    processed_seqs += len(names_batch)
                    progress_ratio = processed_seqs / total_seqs if total_seqs > 0 else 0
                    print(f"[Predict] Progress: {processed_seqs}/{total_seqs} ({progress_ratio*100:.2f}%)", end="\r", flush=True)
                    f.flush()
                    os.fsync(f.fileno())
        print()
        # 正常结束时关闭文件
        class1_f.close()
        class1_fasta_f.close()
    except Exception as e:
        print(f"\n[Predict] Error writing results to file: {e}")
        # 出错时也尽量关闭文件
        try:
            class1_f.close()
        except Exception:
            pass
        try:
            class1_fasta_f.close()
        except Exception:
            pass
    
    # 5. 显示结果
    print("\n--- Prediction Results ---")
    print(f"Total sequences predicted: {len(results)}")
    
    # (可选) 打印前 20 个结果
    for res in results[:20]:
        print(f"\n> {res['name']}")
        print(f"  Predicted Class: {res['predicted_class']}")
        print(f"  Probabilities:   [Class 0: {res['prob_class_0']:.4f}, Class 1: {res['prob_class_1']:.4f}]")
    
    print(f"\n[Predict] Class 1 count: {len(class1_results)}")
    for res in class1_results[:20]:
        print(f"\n> {res['name']}")
        print(f"  Predicted Class: {res['predicted_class']}")
        print(f"  Probabilities:   [Class 0: {res['prob_class_0']:.4f}, Class 1: {res['prob_class_1']:.4f}]")
    
    print(f"\n[Predict] Full results saved (incrementally) to {output_file}")
    print(f"[Predict] Class 1 CSV (with sequences) saved to {class1_file}")
    print(f"[Predict] Class 1 FASTA saved to {class1_fasta_file}")


if __name__ == "__main__":
    main()