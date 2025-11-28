import torch
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify import config, data_loader, model, utils

def predict(input_fasta, output_dir=None):
    # Setup
    log_file = os.path.join(config.LOG_DIR, "predict_log.txt")
    utils.log_message(f"Starting prediction for {input_fasta}", log_file)
    
    device = config.STAGE2_DEVICE
    print(f"[Predict] Using device: {device}")
    
    # 加载模型
    classifier = model.ESMCClassifier(
        embedding_dim=config.EMBEDDING_DIM,
        freeze_base=True # 仅推理
    ).to(device)
    
    model_path = config.PREDICT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[Predict] Error: Model not found at {model_path}")
        return
        
    print(f"[Predict] Loading fine-tuned weights from {model_path}...")
    utils.load_checkpoint(classifier, None, model_path)
    classifier.eval()
    
    # 加载数据
    print(f"[Predict] Loading sequences from {input_fasta}...")
    if not os.path.exists(input_fasta):
        print(f"[Predict] Error: Input FASTA file not found.")
        return
        
    sequences = data_loader.parse_fasta(input_fasta)
    if not sequences:
        print("[Predict] No sequences found.")
        return
        
    seq_list = [s[1] for s in sequences]
    headers = [s[0] for s in sequences]
    
    # 截断序列以防止 OOM
    max_length = 1024
    original_lengths = [len(seq) for seq in seq_list]
    seq_list = [seq[:max_length] for seq in seq_list]
    
    # 报告截断情况
    truncated_count = sum(1 for orig_len in original_lengths if orig_len > max_length)
    if truncated_count > 0:
        print(f"\n[Predict] Truncated {truncated_count} sequences to {max_length} residues to prevent OOM errors.")
        max_orig_len = max(original_lengths)
        print(f"[Predict] Longest original sequence: {max_orig_len} residues")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件
    output_file = os.path.join(output_dir, "predictions_output.csv")
    class1_file = os.path.join(output_dir, "predictions_output_class1.csv")
    class1_fasta_file = os.path.join(output_dir, "predictions_output_class1.fasta")
    
    # 预测
    print("[Predict] Running inference...")
    results = []
    class1_results = []
    embeddings_list = []
    
    batch_size = config.STAGE2_BATCH_SIZE
    total_seqs = len(seq_list)
    processed_seqs = 0
    
    # 打开输出文件
    with open(output_file, 'w') as f, \
         open(class1_file, 'w') as class1_f, \
         open(class1_fasta_file, 'w') as class1_fasta_f:
        
        # 写入表头
        f.write("sequence_name,predicted_class,prob_class_0,prob_class_1\n")
        class1_f.write("prob_class_1,sequence,predicted_class,sequence_name,prob_class_0\n")
        
        with torch.no_grad():
            for i in range(0, len(seq_list), batch_size):
                batch_seqs = seq_list[i:i+batch_size]
                batch_headers = headers[i:i+batch_size]
                
                # 获取 embeddings 和 logits
                batch_emb = classifier.forward_encoder(batch_seqs)
                outputs = classifier.classifier(batch_emb)
                
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # 存储 embeddings 用于可视化
                embeddings_list.append(batch_emb.cpu().numpy())
                
                # 处理结果
                for j in range(len(batch_seqs)):
                    name = batch_headers[j]
                    seq = batch_seqs[j]
                    
                    res = {
                        "name": name,
                        "prob_class_0": probabilities[j, 0].item(),
                        "prob_class_1": probabilities[j, 1].item(),
                        "predicted_class": predictions[j].item(),
                        "sequence": seq
                    }
                    results.append(res)
                    
                    # 写入主 CSV
                    f.write(
                        f"{res['name']},{res['predicted_class']},"
                        f"{res['prob_class_0']:.6f},{res['prob_class_1']:.6f}\n"
                    )
                    
                    # 如果预测为类别 1，写入 class1 文件
                    if res['predicted_class'] == 1:
                        class1_results.append(res)
                        
                        # 类别 1 CSV
                        class1_f.write(
                            f"{res['prob_class_1']:.6f},"
                            f"{res['sequence']},"
                            f"{res['predicted_class']},"
                            f"{res['name']},"
                            f"{res['prob_class_0']:.6f}\n"
                        )
                        
                        # 类别 1 FASTA
                        class1_fasta_f.write(
                            f">{res['name']}\n{res['sequence']}\n"
                        )
                
                processed_seqs += len(batch_seqs)
                progress_ratio = processed_seqs / total_seqs if total_seqs > 0 else 0
                print(f"[Predict] Progress: {processed_seqs}/{total_seqs} ({progress_ratio*100:.2f}%)", end="\r", flush=True)
                
                # 刷新文件
                f.flush()
                os.fsync(f.fileno())
                class1_f.flush()
                os.fsync(class1_f.fileno())
                class1_fasta_f.flush()
                os.fsync(class1_fasta_f.fileno())
    
    print()  # 进度后换行
    
    # 显示结果
    print("\n--- Prediction Results ---")
    print(f"Total sequences predicted: {len(results)}")
    
    # 打印前 20 个结果
    for res in results[:20]:
        print(f"\n> {res['name']}")
        print(f"  Predicted Class: {res['predicted_class']}")
        print(f"  Probabilities:   [Class 0: {res['prob_class_0']:.4f}, Class 1: {res['prob_class_1']:.4f}]")
    
    print(f"\n[Predict] Class 1 count: {len(class1_results)}")
    for res in class1_results[:20]:
        print(f"\n> {res['name']}")
        print(f"  Predicted Class: {res['predicted_class']}")
        print(f"  Probabilities:   [Class 0: {res['prob_class_0']:.4f}, Class 1: {res['prob_class_1']:.4f}]")
    
    print(f"\n[Predict] Full results saved to {output_file}")
    print(f"[Predict] Class 1 CSV (with sequences) saved to {class1_file}")
    print(f"[Predict] Class 1 FASTA saved to {class1_fasta_file}")
    
    # 可视化：训练/测试/预测组合
    print("\n[Predict] Generating embedding visualization...")
    
    # 提取预测用于可视化
    all_predictions = [r['predicted_class'] for r in results]
    
    # 获取训练和测试 embeddings
    print("[Predict] Loading train and test sets for comparison...")
    train_seqs, train_labels, train_ids = data_loader.get_train_sequences_for_visualization(sample_size=2000)
    test_seqs, test_labels, test_ids = data_loader.get_all_sequences_for_visualization()
    
    # 生成训练和测试 embeddings
    classifier.eval()
    with torch.no_grad():
        # 训练 embeddings
        train_embeddings = []
        for i in range(0, len(train_seqs), batch_size):
            batch = train_seqs[i:i+batch_size]
            emb = classifier.forward_encoder(batch)
            train_embeddings.append(emb.cpu().numpy())
        train_embeddings = np.concatenate(train_embeddings, axis=0)
        
        # 测试 embeddings
        test_embeddings = []
        for i in range(0, len(test_seqs), batch_size):
            batch = test_seqs[i:i+batch_size]
            emb = classifier.forward_encoder(batch)
            test_embeddings.append(emb.cpu().numpy())
        test_embeddings = np.concatenate(test_embeddings, axis=0)
    
    # 合并所有 embeddings 进行降维
    all_embeddings = np.concatenate(embeddings_list, axis=0)  # 预测
    combined_embeddings = np.concatenate([train_embeddings, test_embeddings, all_embeddings], axis=0)
    
    # 降维
    print(f"[Predict] Reducing dimensions using {config.VISUALIZATION_METHOD}...")
    if config.VISUALIZATION_METHOD == 'UMAP':
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto')
    
    combined_2d = reducer.fit_transform(combined_embeddings)
    
    # 拆分回原数据
    n_train = len(train_embeddings)
    n_test = len(test_embeddings)
    n_predict = len(all_embeddings)
    
    train_2d = combined_2d[:n_train]
    test_2d = combined_2d[n_train:n_train+n_test]
    predict_2d = combined_2d[n_train+n_test:]
    
    # 保存预测坐标到 CSV
    coords_file = os.path.join(output_dir, "prediction_coordinates.csv")
    print(f"[Predict] Saving coordinates to {coords_file}...")
    import csv
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
    
    # 保存类别 1 坐标到单独 CSV
    class1_coords_file = os.path.join(output_dir, "prediction_class1_coordinates.csv")
    print(f"[Predict] Saving class1 coordinates to {class1_coords_file}...")
    with open(class1_coords_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_name', 'predicted_class', 'prob_class_0', 'prob_class_1', 'x', 'y'])
        for i, res in enumerate(results):
            if res['predicted_class'] == 1:
                writer.writerow([
                    res['name'],
                    res['predicted_class'],
                    f"{res['prob_class_0']:.6f}",
                    f"{res['prob_class_1']:.6f}",
                    predict_2d[i, 0],
                    predict_2d[i, 1]
                ])
    
    # 绘制组合可视化
    vis_path = os.path.join(output_dir, "prediction_vis.png")
    print(f"[Predict] Generating combined visualization...")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 12))
    
    # Plot in order: Predictions (background) -> Train (middle) -> Test (foreground)
    
    # 1. 预测点 (背景, 空心, 低透明度)
    predict_mono_mask = np.array(all_predictions) == 0
    predict_di_mask = np.array(all_predictions) == 1
    plt.scatter(predict_2d[predict_mono_mask, 0], predict_2d[predict_mono_mask, 1],
                facecolors='none', edgecolors='green', label='Predict Mono', alpha=0.3, s=20, marker='o', linewidths=0.8)
    plt.scatter(predict_2d[predict_di_mask, 0], predict_2d[predict_di_mask, 1],
                facecolors='none', edgecolors='orange', label='Predict Di', alpha=0.3, s=20, marker='o', linewidths=0.8)
    
    # 2. Plot Train (middle layer, more visible)
    train_mono_mask = np.array(train_labels) == 0
    train_di_mask = np.array(train_labels) == 1
    plt.scatter(train_2d[train_mono_mask, 0], train_2d[train_mono_mask, 1],
                c='deepskyblue', label='Train Mono', alpha=0.6, s=25, marker='o', edgecolors='blue', linewidths=0.3)
    plt.scatter(train_2d[train_di_mask, 0], train_2d[train_di_mask, 1],
                c='lightcoral', label='Train Di', alpha=0.6, s=25, marker='o', edgecolors='red', linewidths=0.3)
    
    # 3. Plot Test (foreground, most prominent)
    test_mono_mask = np.array(test_labels) == 0
    test_di_mask = np.array(test_labels) == 1
    plt.scatter(test_2d[test_mono_mask, 0], test_2d[test_mono_mask, 1],
                c='blue', label='Test Mono', alpha=0.9, s=50, marker='X', edgecolors='darkblue', linewidths=0.8)
    plt.scatter(test_2d[test_di_mask, 0], test_2d[test_di_mask, 1],
                c='red', label='Test Di', alpha=0.9, s=50, marker='X', edgecolors='darkred', linewidths=0.8)
    
    plt.title(f"Protein Embeddings Visualization ({config.VISUALIZATION_METHOD})\nTest (X) + Train (○) + Predictions (○)", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    
    print(f"[Predict] Visualization saved to {vis_path}")
    
    # 生成聚类边界可视化
    boundary_vis_path = os.path.join(output_dir, "prediction_vis_with_boundaries.png")
    print(f"[Predict] Generating cluster boundary visualization...")
    
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN
    
    plt.figure(figsize=(14, 12))
    
    # 合并训练和测试数据
    train_test_mono_2d = np.vstack([train_2d[train_mono_mask], test_2d[test_mono_mask]])
    train_test_di_2d = np.vstack([train_2d[train_di_mask], test_2d[test_di_mask]])
    
    # 绘制聚类凸包函数
    def draw_cluster_hulls(points, color, label_prefix, min_cluster_size=5):
        """Use DBSCAN to find clusters and draw convex hulls"""
        if len(points) < 3:
            return []
        
        # Use DBSCAN to detect clusters (smaller eps = tighter boundaries)
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
                    # 绘制凸包
                    for simplex in hull.simplices:
                        plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                                color=color, linewidth=2, alpha=0.8)
                    # 填充凸包
                    hull_points = cluster_points[hull.vertices]
                    plt.fill(hull_points[:, 0], hull_points[:, 1], 
                            color=color, alpha=0.15, label=f'{label_prefix} Cluster {cluster_id+1}' if cluster_id == 0 else '')
                    
                    # 存储边界用于放大视图
                    x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
                    y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()
                    cluster_bounds.append((x_min, x_max, y_min, y_max))
                except:
                    pass
        
        return cluster_bounds
    
    # 绘制聚类边界并获取范围
    mono_bounds = draw_cluster_hulls(train_test_mono_2d, 'blue', 'Mono')
    di_bounds = draw_cluster_hulls(train_test_di_2d, 'red', 'Di')
    
    # Plot in order: Predictions (background) -> Train (middle) -> Test (foreground)
    # Same as main vis plot
    
    # 1. 预测点 (背景, 空心, 低透明度)
    plt.scatter(predict_2d[predict_mono_mask, 0], predict_2d[predict_mono_mask, 1],
                facecolors='none', edgecolors='green', label='Predict Mono', alpha=0.3, s=20, marker='o', linewidths=0.8)
    plt.scatter(predict_2d[predict_di_mask, 0], predict_2d[predict_di_mask, 1],
                facecolors='none', edgecolors='orange', label='Predict Di', alpha=0.3, s=20, marker='o', linewidths=0.8)
    
    # 2. Plot Train (middle layer, more visible)
    plt.scatter(train_2d[train_mono_mask, 0], train_2d[train_mono_mask, 1],
                c='deepskyblue', label='Train Mono', alpha=0.6, s=25, marker='o', edgecolors='blue', linewidths=0.3)
    plt.scatter(train_2d[train_di_mask, 0], train_2d[train_di_mask, 1],
                c='lightcoral', label='Train Di', alpha=0.6, s=25, marker='o', edgecolors='red', linewidths=0.3)
    
    # 3. Plot Test (foreground, most prominent)
    plt.scatter(test_2d[test_mono_mask, 0], test_2d[test_mono_mask, 1],
                c='blue', label='Test Mono', alpha=0.9, s=50, marker='X', edgecolors='darkblue', linewidths=0.8)
    plt.scatter(test_2d[test_di_mask, 0], test_2d[test_di_mask, 1],
                c='red', label='Test Di', alpha=0.9, s=50, marker='X', edgecolors='darkred', linewidths=0.8)
    
    plt.title(f"Cluster Boundaries Visualization ({config.VISUALIZATION_METHOD})\nShaded areas show Train+Test clusters", 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(boundary_vis_path, dpi=150)
    plt.close()
    
    print(f"[Predict] Cluster boundary visualization saved to {boundary_vis_path}")
    
    # Generate zoomed-in views for Mono and Di regions
    # Get global axis limits for consistency
    all_points = np.vstack([train_test_mono_2d, train_test_di_2d, predict_2d])
    global_x_min, global_x_max = all_points[:, 0].min(), all_points[:, 0].max()
    global_y_min, global_y_max = all_points[:, 1].min(), all_points[:, 1].max()
    
    # Zoomed view for Mono region
    if mono_bounds:
        mono_zoom_path = os.path.join(output_dir, "prediction_vis_mono_zoom.png")
        print(f"[Predict] Generating Mono region zoom...")
        
        plt.figure(figsize=(12, 12))  # Square figure
        
        # Calculate zoom bounds with padding
        x_mins = [b[0] for b in mono_bounds]
        x_maxs = [b[1] for b in mono_bounds]
        y_mins = [b[2] for b in mono_bounds]
        y_maxs = [b[3] for b in mono_bounds]
        
        x_min, x_max = min(x_mins), max(x_maxs)
        y_min, y_max = min(y_mins), max(y_maxs)
        
        # Add 20% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.2
        x_max += x_range * 0.2
        y_min -= y_range * 0.2
        y_max += y_range * 0.2
        
        # Make it square by using the larger range
        max_range = max(x_range, y_range) * 1.4  # 1.4 = 1.0 + 0.2*2 for padding
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min = x_center - max_range / 2
        x_max = x_center + max_range / 2
        y_min = y_center - max_range / 2
        y_max = y_center + max_range / 2
        
        # Plot all data but zoom to Mono region
        plt.scatter(train_test_mono_2d[:, 0], train_test_mono_2d[:, 1],
                    c='deepskyblue', label='Train+Test Mono', alpha=0.6, s=40, marker='o', edgecolors='blue', linewidths=0.5)
        plt.scatter(train_test_di_2d[:, 0], train_test_di_2d[:, 1],
                    c='lightcoral', label='Train+Test Di', alpha=0.3, s=20, marker='o', edgecolors='red', linewidths=0.3)
        
        # Track if we've added legend labels
        mono_label_added = False
        di_label_added = False
        
        # Highlight predictions in this region
        for i, (x, y) in enumerate(predict_2d):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                pred_class = all_predictions[i]
                
                if pred_class == 0:  # Mono
                    label = 'Predict Mono' if not mono_label_added else ''
                    plt.scatter(x, y, c='green', s=100, marker='^', 
                               edgecolors='darkgreen', linewidths=1.5, zorder=10, label=label)
                    mono_label_added = True
                else:  # Di
                    label = 'Predict Di' if not di_label_added else ''
                    plt.scatter(x, y, c='orange', s=100, marker='^', 
                               edgecolors='darkorange', linewidths=1.5, zorder=10, label=label)
                    di_label_added = True

        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', adjustable='box')  # Force square aspect ratio
        plt.title(f"Mono Region Zoom ({config.VISUALIZATION_METHOD})\nCoordinates consistent with main view", 
                  fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(mono_zoom_path, dpi=150)
        plt.close()
        
        # Save Mono region coordinates to CSV
        mono_coords_file = os.path.join(output_dir, "prediction_mono_region_coordinates.csv")
        mono_fasta_file = os.path.join(output_dir, "prediction_mono_region.fasta")
        
        with open(mono_coords_file, 'w', newline='') as f_csv, \
             open(mono_fasta_file, 'w') as f_fasta:
            writer = csv.writer(f_csv)
            writer.writerow(['sequence_name', 'predicted_class', 'prob_class_0', 'prob_class_1', 'x', 'y'])
            
            for i, (x, y) in enumerate(predict_2d):
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    res = results[i]
                    # Write CSV
                    writer.writerow([
                        res['name'],
                        res['predicted_class'],
                        f"{res['prob_class_0']:.6f}",
                        f"{res['prob_class_1']:.6f}",
                        x,
                        y
                    ])
                    # Write FASTA
                    f_fasta.write(f">{res['name']}\n{res['sequence']}\n")
        
        print(f"[Predict] Mono zoom saved to {mono_zoom_path}")
        print(f"[Predict] Mono region coordinates saved to {mono_coords_file}")
        print(f"[Predict] Mono region FASTA saved to {mono_fasta_file}")
    
    # Zoomed view for Di region
    if di_bounds:
        di_zoom_path = os.path.join(output_dir, "prediction_vis_di_zoom.png")
        print(f"[Predict] Generating Di region zoom...")
        
        plt.figure(figsize=(12, 12))  # Square figure
        
        # Calculate zoom bounds with padding
        x_mins = [b[0] for b in di_bounds]
        x_maxs = [b[1] for b in di_bounds]
        y_mins = [b[2] for b in di_bounds]
        y_maxs = [b[3] for b in di_bounds]
        
        x_min, x_max = min(x_mins), max(x_maxs)
        y_min, y_max = min(y_mins), max(y_maxs)
        
        # Add 20% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.2
        x_max += x_range * 0.2
        y_min -= y_range * 0.2
        y_max += y_range * 0.2
        
        # Make it square by using the larger range
        max_range = max(x_range, y_range) * 1.4  # 1.4 = 1.0 + 0.2*2 for padding
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min = x_center - max_range / 2
        x_max = x_center + max_range / 2
        y_min = y_center - max_range / 2
        y_max = y_center + max_range / 2
        
        # Plot all data but zoom to Di region
        plt.scatter(train_test_mono_2d[:, 0], train_test_mono_2d[:, 1],
                    c='deepskyblue', label='Train+Test Mono', alpha=0.3, s=20, marker='o', edgecolors='blue', linewidths=0.3)
        plt.scatter(train_test_di_2d[:, 0], train_test_di_2d[:, 1],
                    c='lightcoral', label='Train+Test Di', alpha=0.6, s=40, marker='o', edgecolors='red', linewidths=0.5)
        
        # Track if we've added legend labels
        mono_label_added = False
        di_label_added = False
        
        # Highlight predictions in this region
        for i, (x, y) in enumerate(predict_2d):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                pred_class = all_predictions[i]
                
                if pred_class == 0:  # Mono
                    label = 'Predict Mono' if not mono_label_added else ''
                    plt.scatter(x, y, c='green', s=100, marker='^', 
                               edgecolors='darkgreen', linewidths=1.5, zorder=10, label=label)
                    mono_label_added = True
                else:  # Di
                    label = 'Predict Di' if not di_label_added else ''
                    plt.scatter(x, y, c='orange', s=100, marker='^', 
                               edgecolors='darkorange', linewidths=1.5, zorder=10, label=label)
                    di_label_added = True

        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', adjustable='box')  # Force square aspect ratio
        plt.title(f"Di Region Zoom ({config.VISUALIZATION_METHOD})\nCoordinates consistent with main view", 
                  fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(di_zoom_path, dpi=150)
        plt.close()
        
        # Save Di region coordinates to CSV
        di_coords_file = os.path.join(output_dir, "prediction_di_region_coordinates.csv")
        di_fasta_file = os.path.join(output_dir, "prediction_di_region.fasta")
        
        with open(di_coords_file, 'w', newline='') as f_csv, \
             open(di_fasta_file, 'w') as f_fasta:
            writer = csv.writer(f_csv)
            writer.writerow(['sequence_name', 'predicted_class', 'prob_class_0', 'prob_class_1', 'x', 'y'])
            
            for i, (x, y) in enumerate(predict_2d):
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    res = results[i]
                    # Write CSV
                    writer.writerow([
                        res['name'],
                        res['predicted_class'],
                        f"{res['prob_class_0']:.6f}",
                        f"{res['prob_class_1']:.6f}",
                        x,
                        y
                    ])
                    # Write FASTA
                    f_fasta.write(f">{res['name']}\n{res['sequence']}\n")
        
        print(f"[Predict] Di zoom saved to {di_zoom_path}")
        print(f"[Predict] Di region coordinates saved to {di_coords_file}")
        print(f"[Predict] Di region FASTA saved to {di_fasta_file}")
    
    print(f"[Predict] Coordinates saved to {coords_file}")
    
    utils.log_message(f"Prediction complete. {len(results)} sequences processed.", log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict protein classes using ESM-C")
    parser.add_argument("input_fasta", help="Path to input FASTA file")
    parser.add_argument("--output_dir", help="Path to output directory", default=None)
    
    args = parser.parse_args()
    predict(args.input_fasta, args.output_dir)
