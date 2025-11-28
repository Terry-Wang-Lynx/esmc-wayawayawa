import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

# --- 路径修复 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ----------------

from model import get_model_and_tokenizer
# 导入 FastaDataset (用于读取) 和 PredictCollate (用于打包)
from data_loader import FastaDataset, PredictCollate, TrainCollate

# --- 配置参数 ---
MODEL_NAME = "esmc_600m"
# ! 你的本地权重路径 (请修改)
LOCAL_WEIGHTS_PATH = "/home/wangty/esm/esm/data/weights/esmc_600m_2024_12_v0.pth"
NUM_CLASSES = 2

# ! 加载我们微调过的权重
MODEL_PATH = "/home/wangty/esm/esm/discovery/outputs/TIM-barrel/weights/parameter/esmc_classifier_epoch_80.pth"

# ! 需要预测的 FASTA 文件
FASTA_TO_PREDICT = "/home/wangty/esm/esm/discovery/datasets/all_archaea_proteins.fasta"

# ! 输出目录
OUTPUT_DIR = "/home/wangty/esm/esm/discovery/inference/TIM-barrel"

# ! 训练集和测试集路径 (用于可视化背景)
TRAIN_POS_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "train_positive.fasta")
TRAIN_NEG_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "train_negative.fasta")
TEST_POS_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "test_positive.fasta")
TEST_NEG_FASTA = os.path.join(SCRIPT_DIR, "datasets", "TIM-barrel", "test_negative.fasta")

# 预测时的批次大小
BATCH_SIZE = 16 
# 从中间开始预测的起始序列编号（从 0 开始计数），例如 5000 表示跳过前 5000 条序列
START_INDEX = 0

# 可视化方法: 'UMAP' or 't-SNE'
VISUALIZATION_METHOD = 'UMAP'
# -----------------

def get_embeddings(model, data_loader, device):
    """
    提取数据集的 embeddings
    """
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            # PredictCollate 返回 (names, tokens, seqs)
            # TrainCollate 返回 (tokens, labels)
            # FastaDataset[index] 返回 (name, seq) or ((name, seq), label)
            
            # 这里我们需要处理两种不同的 Collate 输出，或者统一使用一种方式
            # 为了简单，我们假设这里传入的 data_loader 使用的是 PredictCollate 或者类似的结构
            # 但是 TrainingDataset 使用的是 TrainCollate，返回 (tokens, labels)
            
            if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[1], torch.Tensor):
                 # TrainCollate: (tokens, labels)
                tokens = batch[0]
            elif isinstance(batch, tuple) and len(batch) == 3:
                # PredictCollate: (names, tokens, seqs)
                tokens = batch[1]
            else:
                # 尝试直接取第一个元素作为 tokens
                tokens = batch[0]

            tokens = tokens.to(device)
            # 假设 model 是 ESMCClassifier 实例
            if hasattr(model, 'forward_encoder'):
                emb = model.forward_encoder(tokens)
            else:
                # Fallback: 如果没有 forward_encoder，可能 model 就是直接返回 logits
                # 这种情况下很难拿到 embedding，除非修改 model 定义
                # 假设我们用的是和 classify 一样的 model.py
                # 暂时用 logits 代替 embedding (不推荐) 或者报错
                # 为了代码健壮性，这里先占位，如果报错则需要检查 model.py
                emb = torch.zeros((tokens.size(0), 1024)).to(device) 
                # print("[Warning] Model does not have forward_encoder, visualization might be incorrect.")
            
            embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
    embeddings_list = [] # 用于存储预测数据的 embeddings
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(OUTPUT_DIR, "predictions_output.csv")
    class1_file = os.path.join(OUTPUT_DIR, "predictions_output_class1.csv")
    class1_fasta_file = os.path.join(OUTPUT_DIR, "predictions_output_class1.fasta")

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
    
                    # 获取 embeddings 和 logits
                    # 假设 model 是 ESMCClassifier 实例
                    if hasattr(model, 'forward_encoder'):
                        batch_emb = model.forward_encoder(batch_tokens)
                        logits = model.classifier(batch_emb)
                    else:
                        # Fallback
                        logits = model(batch_tokens)
                        batch_emb = torch.zeros((len(names_batch), 1024)).to(device) 
                        # print("[Warning] Model does not have forward_encoder, visualization might be incorrect.")

                    embeddings_list.append(batch_emb.cpu().numpy())

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
    
    print(f"\n[Predict] Full results saved (incrementally) to {output_file}")
    print(f"[Predict] Class 1 CSV (with sequences) saved to {class1_file}")
    print(f"[Predict] Class 1 FASTA saved to {class1_fasta_file}")


    # --- 6. 可视化部分 ---
    print("\n[Predict] Generating embedding visualization...")

    # 6.1 加载训练集和测试集作为背景
    print("[Predict] Loading train and test sets for comparison...")
    
    # 辅助函数：加载数据并提取 embeddings
    def load_and_embed(fasta_path, label, sample_size=None):
        if not os.path.exists(fasta_path):
            print(f"[Warning] File not found: {fasta_path}")
            return np.array([]), np.array([])
        
        dataset = FastaDataset(fasta_path, label=label)
        # 如果需要采样
        if sample_size and len(dataset) > sample_size:
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
            
        # 使用 TrainCollate 来获取 tokens (因为 dataset 返回 ((name, seq), label))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TrainCollate(tokenizer), num_workers=0)
        
        embs = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                # TrainCollate returns (tokens, labels)
                tokens = batch[0]
                tokens = tokens.to(device)
                if hasattr(model, 'forward_encoder'):
                    emb = model.forward_encoder(tokens)
                else:
                    emb = model(tokens) # Fallback
                embs.append(emb.cpu().numpy())
        
        if not embs:
            return np.array([]), np.array([])
            
        embs = np.concatenate(embs, axis=0)
        labels = np.full(len(embs), label)
        return embs, labels

    # 加载训练集 (采样以避免点太多)
    train_pos_emb, train_pos_lbl = load_and_embed(TRAIN_POS_FASTA, 1, sample_size=1000)
    train_neg_emb, train_neg_lbl = load_and_embed(TRAIN_NEG_FASTA, 0, sample_size=1000)
    
    # 加载测试集
    test_pos_emb, test_pos_lbl = load_and_embed(TEST_POS_FASTA, 1)
    test_neg_emb, test_neg_lbl = load_and_embed(TEST_NEG_FASTA, 0)
    
    # 合并训练和测试数据
    train_embeddings = np.concatenate([train_pos_emb, train_neg_emb], axis=0) if len(train_pos_emb) > 0 or len(train_neg_emb) > 0 else np.array([])
    train_labels = np.concatenate([train_pos_lbl, train_neg_lbl], axis=0) if len(train_pos_lbl) > 0 or len(train_neg_lbl) > 0 else np.array([])
    
    test_embeddings = np.concatenate([test_pos_emb, test_neg_emb], axis=0) if len(test_pos_emb) > 0 or len(test_neg_emb) > 0 else np.array([])
    test_labels = np.concatenate([test_pos_lbl, test_neg_lbl], axis=0) if len(test_pos_lbl) > 0 or len(test_neg_lbl) > 0 else np.array([])

    # 预测数据
    predict_embeddings = np.concatenate(embeddings_list, axis=0)
    predict_labels = np.array([r['predicted_class'] for r in results])
    
    # 6.2 降维
    # 合并所有数据进行降维
    all_embeddings_list = []
    if len(train_embeddings) > 0: all_embeddings_list.append(train_embeddings)
    if len(test_embeddings) > 0: all_embeddings_list.append(test_embeddings)
    all_embeddings_list.append(predict_embeddings)
    
    combined_embeddings = np.concatenate(all_embeddings_list, axis=0)
    
    print(f"[Predict] Reducing dimensions using {VISUALIZATION_METHOD}...")
    if VISUALIZATION_METHOD == 'UMAP':
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
    
    combined_2d = reducer.fit_transform(combined_embeddings)
    
    # 拆分回各个集合
    idx = 0
    train_2d = None
    test_2d = None
    predict_2d = None
    
    if len(train_embeddings) > 0:
        train_2d = combined_2d[idx : idx + len(train_embeddings)]
        idx += len(train_embeddings)
        
    if len(test_embeddings) > 0:
        test_2d = combined_2d[idx : idx + len(test_embeddings)]
        idx += len(test_embeddings)
        
    predict_2d = combined_2d[idx:]
    
    # 6.3 保存坐标
    coords_file = os.path.join(OUTPUT_DIR, "prediction_coordinates.csv")
    print(f"[Predict] Saving coordinates to {coords_file}...")
    with open(coords_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_name', 'predicted_class', 'prob_class_0', 'prob_class_1', 'x', 'y'])
        for i, res in enumerate(results):
            writer.writerow([
                res['name'],
                res['predicted_class'],
                f"{res['prob_class_0']:.6f}",
                f"{res['prob_class_1']:.6f}",
                predict_2d[i, 0],
                predict_2d[i, 1]
            ])

    # 6.4 绘图
    vis_path = os.path.join(OUTPUT_DIR, "prediction_vis.png")
    print(f"[Predict] Generating combined visualization...")
    
    plt.figure(figsize=(14, 12))
    
    # 1. Plot Predictions (background)
    predict_mono_mask = predict_labels == 0
    predict_di_mask = predict_labels == 1
    plt.scatter(predict_2d[predict_mono_mask, 0], predict_2d[predict_mono_mask, 1],
                c='lightgreen', label='Predict Mono', alpha=0.4, s=30, marker='o', edgecolors='green', linewidths=0.5)
    plt.scatter(predict_2d[predict_di_mask, 0], predict_2d[predict_di_mask, 1],
                c='lightsalmon', label='Predict Di', alpha=0.4, s=30, marker='o', edgecolors='orange', linewidths=0.5)
    
    # 2. Plot Train (middle)
    if train_2d is not None:
        train_mono_mask = train_labels == 0
        train_di_mask = train_labels == 1
        plt.scatter(train_2d[train_mono_mask, 0], train_2d[train_mono_mask, 1],
                    c='deepskyblue', label='Train Mono', alpha=0.6, s=25, marker='o', edgecolors='blue', linewidths=0.3)
        plt.scatter(train_2d[train_di_mask, 0], train_2d[train_di_mask, 1],
                    c='lightcoral', label='Train Di', alpha=0.6, s=25, marker='o', edgecolors='red', linewidths=0.3)
    
    # 3. Plot Test (foreground)
    if test_2d is not None:
        test_mono_mask = test_labels == 0
        test_di_mask = test_labels == 1
        plt.scatter(test_2d[test_mono_mask, 0], test_2d[test_mono_mask, 1],
                    c='blue', label='Test Mono', alpha=0.9, s=50, marker='X', edgecolors='darkblue', linewidths=0.8)
        plt.scatter(test_2d[test_di_mask, 0], test_2d[test_di_mask, 1],
                    c='red', label='Test Di', alpha=0.9, s=50, marker='X', edgecolors='darkred', linewidths=0.8)
    
    plt.title(f"Protein Embeddings Visualization ({VISUALIZATION_METHOD})\nTest (X) + Train (o) + Predictions (o)", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"[Predict] Visualization saved to {vis_path}")

    # 6.5 聚类边界和放大图
    # 组合 Train + Test 用于计算边界
    train_test_mono_2d = []
    train_test_di_2d = []
    
    if train_2d is not None:
        train_test_mono_2d.append(train_2d[train_mono_mask])
        train_test_di_2d.append(train_2d[train_di_mask])
    if test_2d is not None:
        train_test_mono_2d.append(test_2d[test_mono_mask])
        train_test_di_2d.append(test_2d[test_di_mask])
        
    train_test_mono_2d = np.vstack(train_test_mono_2d) if train_test_mono_2d else np.array([])
    train_test_di_2d = np.vstack(train_test_di_2d) if train_test_di_2d else np.array([])

    # Function to draw convex hulls for clusters
    def draw_cluster_hulls(points, color, label_prefix, min_cluster_size=5):
        if len(points) < 3:
            return []
        
        clustering = DBSCAN(eps=1.8, min_samples=min_cluster_size).fit(points)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_bounds = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    # Store bounds
                    x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
                    y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()
                    cluster_bounds.append((x_min, x_max, y_min, y_max))
                except:
                    pass
        return cluster_bounds

    # 计算边界
    mono_bounds = draw_cluster_hulls(train_test_mono_2d, 'blue', 'Mono') if len(train_test_mono_2d) > 0 else []
    di_bounds = draw_cluster_hulls(train_test_di_2d, 'red', 'Di') if len(train_test_di_2d) > 0 else []

    # 生成放大图 (Mono Region)
    if mono_bounds:
        mono_zoom_path = os.path.join(OUTPUT_DIR, "prediction_vis_mono_zoom.png")
        print(f"[Predict] Generating Mono region zoom...")
        
        plt.figure(figsize=(12, 12))
        
        # Calculate zoom bounds
        x_mins = [b[0] for b in mono_bounds]
        x_maxs = [b[1] for b in mono_bounds]
        y_mins = [b[2] for b in mono_bounds]
        y_maxs = [b[3] for b in mono_bounds]
        
        x_min, x_max = min(x_mins), max(x_maxs)
        y_min, y_max = min(y_mins), max(y_maxs)
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.2
        x_max += x_range * 0.2
        y_min -= y_range * 0.2
        y_max += y_range * 0.2
        
        # Square aspect
        max_range = max(x_range, y_range) * 1.4
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min = x_center - max_range / 2
        x_max = x_center + max_range / 2
        y_min = y_center - max_range / 2
        y_max = y_center + max_range / 2
        
        # Plot background (Train/Test)
        if len(train_test_mono_2d) > 0:
            plt.scatter(train_test_mono_2d[:, 0], train_test_mono_2d[:, 1],
                        c='deepskyblue', label='Train+Test Mono', alpha=0.6, s=40, marker='o', edgecolors='blue', linewidths=0.5)
        if len(train_test_di_2d) > 0:
            plt.scatter(train_test_di_2d[:, 0], train_test_di_2d[:, 1],
                        c='lightcoral', label='Train+Test Di', alpha=0.3, s=20, marker='o', edgecolors='red', linewidths=0.3)
        
        # Highlight predictions
        mono_label_added = False
        di_label_added = False
        
        for i, (x, y) in enumerate(predict_2d):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                pred_class = predict_labels[i]
                if pred_class == 0: # Mono
                    label = 'Predict Mono' if not mono_label_added else ''
                    plt.scatter(x, y, c='green', s=100, marker='^', edgecolors='darkgreen', linewidths=1.5, zorder=10, label=label)
                    mono_label_added = True
                else: # Di
                    label = 'Predict Di' if not di_label_added else ''
                    plt.scatter(x, y, c='orange', s=100, marker='^', edgecolors='darkorange', linewidths=1.5, zorder=10, label=label)
                    di_label_added = True
                
                # Annotate
                plt.annotate(results[i]['name'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Mono Region Zoom ({VISUALIZATION_METHOD})", fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(mono_zoom_path, dpi=150)
        plt.close()
        print(f"[Predict] Mono zoom saved to {mono_zoom_path}")

    # 生成放大图 (Di Region)
    if di_bounds:
        di_zoom_path = os.path.join(OUTPUT_DIR, "prediction_vis_di_zoom.png")
        print(f"[Predict] Generating Di region zoom...")
        
        plt.figure(figsize=(12, 12))
        
        x_mins = [b[0] for b in di_bounds]
        x_maxs = [b[1] for b in di_bounds]
        y_mins = [b[2] for b in di_bounds]
        y_maxs = [b[3] for b in di_bounds]
        
        x_min, x_max = min(x_mins), max(x_maxs)
        y_min, y_max = min(y_mins), max(y_maxs)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.2
        x_max += x_range * 0.2
        y_min -= y_range * 0.2
        y_max += y_range * 0.2
        
        max_range = max(x_range, y_range) * 1.4
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min = x_center - max_range / 2
        x_max = x_center + max_range / 2
        y_min = y_center - max_range / 2
        y_max = y_center + max_range / 2
        
        if len(train_test_mono_2d) > 0:
            plt.scatter(train_test_mono_2d[:, 0], train_test_mono_2d[:, 1],
                        c='deepskyblue', label='Train+Test Mono', alpha=0.3, s=20, marker='o', edgecolors='blue', linewidths=0.3)
        if len(train_test_di_2d) > 0:
            plt.scatter(train_test_di_2d[:, 0], train_test_di_2d[:, 1],
                        c='lightcoral', label='Train+Test Di', alpha=0.6, s=40, marker='o', edgecolors='red', linewidths=0.5)
        
        mono_label_added = False
        di_label_added = False
        
        for i, (x, y) in enumerate(predict_2d):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                pred_class = predict_labels[i]
                if pred_class == 0:
                    label = 'Predict Mono' if not mono_label_added else ''
                    plt.scatter(x, y, c='green', s=100, marker='^', edgecolors='darkgreen', linewidths=1.5, zorder=10, label=label)
                    mono_label_added = True
                else:
                    label = 'Predict Di' if not di_label_added else ''
                    plt.scatter(x, y, c='orange', s=100, marker='^', edgecolors='darkorange', linewidths=1.5, zorder=10, label=label)
                    di_label_added = True
                plt.annotate(results[i]['name'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Di Region Zoom ({VISUALIZATION_METHOD})", fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(di_zoom_path, dpi=150)
        plt.close()
        print(f"[Predict] Di zoom saved to {di_zoom_path}")


if __name__ == "__main__":
    main()