import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify import config, data_loader, model, utils

def train_stage2():
    # Setup
    os.makedirs(config.STAGE2_LOG_DIR, exist_ok=True)
    os.makedirs(config.STAGE2_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.STAGE2_VISUALIZATION_DIR, exist_ok=True)
    
    log_file = os.path.join(config.STAGE2_LOG_DIR, "stage2_log.txt")
    utils.log_message("Starting Stage 2: Classification Finetuning", log_file)
    
    device = config.STAGE2_DEVICE
    utils.log_message(f"Using device: {device}", log_file)
    
    # Model
    # 初始化模型
    classifier = model.ESMCClassifier(
        embedding_dim=config.EMBEDDING_DIM,
        freeze_base=config.FREEZE_BASE_MODEL,
        unfreeze_last_n=config.UNFREEZE_LAST_N_LAYERS
    ).to(device)
    
    # 加载 Stage 1 权重
    stage1_weights = config.STAGE2_PRETRAINED_WEIGHTS
    if stage1_weights and os.path.exists(stage1_weights):
        utils.log_message(f"Loading Stage 1 weights from {stage1_weights}", log_file)
        # 只加载编码器部分
        # Stage 1 保存了完整模型
        # Stage 1 使用相同架构
        # Stage 1 只训练了编码器
        # Stage 1 的分类头未训练
        # 因此可以加载编码器权重
        
        checkpoint = torch.load(stage1_weights, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # 过滤分类器键
        # but actually we want to keep the encoder.
        # 键名格式为 esmc... 和 classifier...
        # 应加载 esmc... 键
        
        model_dict = classifier.state_dict()
        # 1. 过滤不必要的键
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and k.startswith('esmc')}
        # 2. 覆盖现有状态字典
        model_dict.update(pretrained_dict) 
        # 3. 加载新状态字典
        classifier.load_state_dict(model_dict)
        
        utils.log_message("Stage 1 encoder weights loaded.", log_file)
    else:
        utils.log_message("Warning: Stage 1 weights not found. Training from scratch/pretrained base.", log_file)
    
    # 重新应用冻结逻辑
    classifier._setup_freezing()
    classifier.unfreeze_layers() # Apply unfreezing logic
    
    # Optimizer
    # 只优化需要梯度的参数
    params_to_optimize = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=config.STAGE2_LEARNING_RATE)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Data
    train_loader = data_loader.get_classification_dataloader(batch_size=config.STAGE2_BATCH_SIZE, is_train=True)
    test_loader = data_loader.get_classification_dataloader(batch_size=config.STAGE2_BATCH_SIZE, is_train=False)
    
    # 指标历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'epoch': [],
        'test_epoch': [] # 记录测试周期
    }

    # Helper function for visualization
    def run_visualization(epoch_num):
        utils.log_message(f"Generating visualization for Epoch {epoch_num}...", log_file)
        classifier.eval()
        try:
            with torch.no_grad():
                # 1. Test Set
                test_seqs, test_labels, test_ids = data_loader.get_all_sequences_for_visualization()
                test_embeddings = []
                batch_size = config.STAGE2_BATCH_SIZE
                for j in range(0, len(test_seqs), batch_size):
                    batch_seqs = test_seqs[j:j+batch_size]
                    batch_emb = classifier.forward_encoder(batch_seqs)
                    test_embeddings.append(batch_emb.cpu().numpy())
                test_embeddings = np.concatenate(test_embeddings, axis=0)
                
                # 2. Train Set (Sampled)
                train_seqs, train_labels, train_ids = data_loader.get_train_sequences_for_visualization(sample_size=2000)
                train_embeddings = []
                for j in range(0, len(train_seqs), batch_size):
                    batch_seqs = train_seqs[j:j+batch_size]
                    batch_emb = classifier.forward_encoder(batch_seqs)
                    train_embeddings.append(batch_emb.cpu().numpy())
                train_embeddings = np.concatenate(train_embeddings, axis=0)
                
                vis_path = os.path.join(config.STAGE2_VISUALIZATION_DIR, f"stage2_epoch_{epoch_num}.png")
                utils.reduce_and_plot_train_test_embeddings(
                    train_embeddings, train_labels,
                    test_embeddings, test_labels,
                    vis_path, 
                    method=config.VISUALIZATION_METHOD,
                    train_ids=train_ids,
                    test_ids=test_ids
                )
                utils.log_message(f"Visualization saved to {vis_path}", log_file)
        except Exception as e:
            utils.log_message(f"Visualization failed: {e}", log_file)
            import traceback
            traceback.print_exc()

    # Initial Visualization (Epoch 0)
    run_visualization(0)
    
    # Training Loop
    for epoch in range(config.STAGE2_EPOCHS):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (seqs, labels) in enumerate(train_loader):
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = classifier(seqs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 实时步骤输出
            if (i + 1) % 10 == 0:
                batch_acc = (predicted == labels).float().mean().item() if labels.size(0) > 0 else 0.0
                avg_loss = total_loss / (i + 1)
                print(f"  Epoch {epoch+1}/{config.STAGE2_EPOCHS}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Batch Acc: {batch_acc * 100:.2f}%")
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"--- Epoch {epoch+1} Complete. Avg Train Loss: {avg_loss:.4f}, Avg Train Acc: {accuracy:.2f}% ---")
        utils.log_message(f"Epoch [{epoch+1}/{config.STAGE2_EPOCHS}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%", log_file)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(accuracy)
        history['epoch'].append(epoch + 1)
        
        # 初始化本轮测试指标
        test_acc = None
        precision = None
        recall = None
        f1 = None

        # 评估
        if (epoch + 1) % config.STAGE2_EVAL_INTERVAL == 0:
            classifier.eval()
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for seqs, labels in test_loader:
                    labels = labels.to(device)
                    outputs = classifier(seqs)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy()) # 正类概率
            
            test_acc = 100 * test_correct / test_total
            
            # 计算详细指标
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            utils.log_message(
                f"Test Evaluation - Acc: {test_acc:.2f}%, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}", 
                log_file
            )
            
            history['test_acc'].append(test_acc)
            history['test_precision'].append(precision)
            history['test_recall'].append(recall)
            history['test_f1'].append(f1)
            history['test_epoch'].append(epoch + 1) # 记录当前周期
            
            # 保存训练曲线
            if (epoch + 1) % config.STAGE2_SAVE_PLOTS_INTERVAL == 0:
                utils.save_training_plots(history, config.STAGE2_OUTPUT_DIR)
            
            # 绘制混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            cm_path = os.path.join(config.STAGE2_VISUALIZATION_DIR, f"confusion_matrix_epoch_{epoch+1}.png")
            utils.plot_confusion_matrix(cm, classes=['Mono', 'Di'], output_path=cm_path)
            
            # 绘制 ROC 和 PR 曲线
            roc_path = os.path.join(config.STAGE2_VISUALIZATION_DIR, f"roc_curve_epoch_{epoch+1}.png")
            utils.plot_roc_curve(all_labels, all_probs, roc_path)
            
            pr_path = os.path.join(config.STAGE2_VISUALIZATION_DIR, f"pr_curve_epoch_{epoch+1}.png")
            utils.plot_precision_recall_curve(all_labels, all_probs, pr_path)

        # 保存指标到 CSV
        csv_path = os.path.join(config.STAGE2_LOG_DIR, "stage2_metrics.csv")
        metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_acc': accuracy,
            'test_acc': test_acc if test_acc is not None else '',
            'test_precision': precision if precision is not None else '',
            'test_recall': recall if recall is not None else '',
            'test_f1': f1 if f1 is not None else ''
        }
        utils.save_metrics_to_csv(metrics, csv_path)

            
        # 可视化
        if (epoch + 1) % config.STAGE2_VISUALIZATION_INTERVAL == 0:
            run_visualization(epoch + 1)
        
        # 定期保存检查点
        if (epoch + 1) % config.STAGE2_SAVE_CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(config.STAGE2_CHECKPOINT_DIR, f"stage2_epoch_{epoch+1}.pth")
            utils.save_checkpoint(classifier, optimizer, epoch, ckpt_path)
            utils.log_message(f"Checkpoint saved to {ckpt_path}", log_file)
                
    # 保存最终模型
    final_path = os.path.join(config.STAGE2_CHECKPOINT_DIR, "stage2_final.pth")
    utils.save_checkpoint(classifier, optimizer, config.STAGE2_EPOCHS, final_path)
    utils.log_message("Stage 2 training complete.", log_file)

if __name__ == "__main__":
    train_stage2()
