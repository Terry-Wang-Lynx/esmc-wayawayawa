import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify import config, data_loader, model, utils

def train_stage1():
    # Setup
    os.makedirs(config.STAGE1_LOG_DIR, exist_ok=True)
    os.makedirs(config.STAGE1_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.STAGE1_VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(config.STAGE1_VISUALIZATION_PLOTS_DIR, exist_ok=True)
    os.makedirs(config.STAGE1_VISUALIZATION_COORDS_DIR, exist_ok=True)
    
    log_file = os.path.join(config.STAGE1_LOG_DIR, "stage1_log.txt")
    utils.log_message("Starting Stage 1: Contrastive Pretraining", log_file)
    
    device = config.STAGE1_DEVICE
    utils.log_message(f"Using device: {device}", log_file)
    
    # Model
    # For Stage 1, we want to fine-tune the encoder.
    # So we set freeze_base=False to allow gradients.
    # Or we can use the config setting if it was meant for both. 
    # But config says FREEZE_BASE_MODEL is for Stage 2.
    # Let's assume we unfreeze everything for Stage 1 or follow a specific strategy.
    # "Fine-tune ESM-C encoder" implies training it.
    utils.log_message("Initializing model...", log_file)
    classifier = model.ESMCClassifier(
        embedding_dim=config.EMBEDDING_DIM,
        freeze_base=True,
        unfreeze_last_n=2
    ).to(device)
    utils.log_message("Model initialized and moved to device.", log_file)
    
    # Optimizer
    optimizer = optim.AdamW(classifier.parameters(), lr=config.STAGE1_LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # Loss
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    
    # Data
    dataloader = data_loader.get_contrastive_dataloader(batch_size=config.STAGE1_BATCH_SIZE)
    
    # Resume
    start_epoch = 0
    if config.RESUME_FROM_CHECKPOINT:
        start_epoch = utils.load_checkpoint(classifier, optimizer, config.RESUME_FROM_CHECKPOINT)
        
    # History tracking
    history = {
        'train_loss': [],
        'pos_sim': [],
        'neg_sim': [],
        'epoch': []
    }
        
    # Training Loop
    for epoch in range(start_epoch, config.STAGE1_EPOCHS):
        classifier.train()
        total_loss = 0
        total_pos_sim = 0
        total_neg_sim = 0
        pos_count = 0
        neg_count = 0
        steps = 0
        
        for i, (seq1, seq2, target) in enumerate(dataloader):
            # Tokenize and move to device
            target = target.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                emb1 = classifier.forward_encoder(seq1)
                emb2 = classifier.forward_encoder(seq2)
                loss = criterion(emb1, emb2, target)
                
                # Calculate metrics
                # We detach to avoid graph retention
                cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).detach()
                
                # Positive pairs (target == 1)
                pos_mask = (target == 1)
                if pos_mask.sum() > 0:
                    total_pos_sim += cos_sim[pos_mask].sum().item()
                    pos_count += pos_mask.sum().item()
                    
                # Negative pairs (target == -1)
                neg_mask = (target == -1)
                if neg_mask.sum() > 0:
                    total_neg_sim += cos_sim[neg_mask].sum().item()
                    neg_count += neg_mask.sum().item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            steps += 1
            
            # Real-time step output (every 10 steps)
            if (i + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )
                
        # Epoch summary
        avg_epoch_loss = total_loss / steps if steps > 0 else 0
        avg_pos_sim = total_pos_sim / pos_count if pos_count > 0 else 0
        avg_neg_sim = total_neg_sim / neg_count if neg_count > 0 else 0
        
        print(
            f"--- Epoch {epoch+1} Complete. "
            f"Loss: {avg_epoch_loss:.4f}, "
            f"Pos Sim: {avg_pos_sim:.4f}, Neg Sim: {avg_neg_sim:.4f} ---"
        )
        utils.log_message(
            f"Epoch {epoch+1} Complete. Loss: {avg_epoch_loss:.4f}, "
            f"Pos Sim: {avg_pos_sim:.4f}, Neg Sim: {avg_neg_sim:.4f}", 
            log_file
        )
        
        # Save metrics to CSV
        csv_path = os.path.join(config.STAGE1_LOG_DIR, "stage1_metrics.csv")
        metrics = {
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'pos_similarity': avg_pos_sim,
            'neg_similarity': avg_neg_sim
        }
        utils.save_metrics_to_csv(metrics, csv_path)
        
        # Update history
        history['train_loss'].append(avg_epoch_loss)
        history['pos_sim'].append(avg_pos_sim)
        history['neg_sim'].append(avg_neg_sim)
        history['epoch'].append(epoch + 1)
        
        # Save plots
        utils.save_training_plots(history, config.STAGE1_OUTPUT_DIR)
                
        # Checkpoint
        if (epoch + 1) % config.SAVE_CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(config.STAGE1_CHECKPOINT_DIR, f"stage1_epoch_{epoch+1}.pth")
            utils.save_checkpoint(classifier, optimizer, epoch, ckpt_path)
            
        # Visualization
        if (epoch + 1) % config.VISUALIZATION_EPOCH_INTERVAL == 0:
            utils.log_message("Generating visualization...", log_file)
            classifier.eval()
            with torch.no_grad():
                # 1. Test Set
                test_seqs, test_labels, test_ids = data_loader.get_all_sequences_for_visualization()
                test_embeddings = []
                batch_size = config.STAGE1_BATCH_SIZE
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
                
                vis_path = os.path.join(config.STAGE1_VISUALIZATION_PLOTS_DIR, f"stage1_epoch_{epoch+1}.png")
                utils.reduce_and_plot_train_test_embeddings(
                    train_embeddings, train_labels,
                    test_embeddings, test_labels,
                    vis_path, 
                    method=config.VISUALIZATION_METHOD,
                    train_ids=train_ids,
                    test_ids=test_ids
                )
    
    # Save final model
    final_path = os.path.join(config.STAGE1_CHECKPOINT_DIR, "stage1_final.pth")
    utils.save_checkpoint(classifier, optimizer, config.STAGE1_EPOCHS, final_path)
    utils.log_message("Stage 1 training complete.", log_file)

if __name__ == "__main__":
    train_stage1()
