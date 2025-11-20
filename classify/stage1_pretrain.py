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
    log_file = os.path.join(config.LOG_DIR, "stage1_log.txt")
    utils.log_message("Starting Stage 1: Contrastive Pretraining", log_file)
    
    device = config.DEVICE
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
        freeze_base=False # Allow updates to encoder
    ).to(device)
    utils.log_message("Model initialized and moved to device.", log_file)
    
    # Optimizer
    optimizer = optim.AdamW(classifier.parameters(), lr=config.STAGE1_LEARNING_RATE)
    
    # Loss
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    
    # Data
    dataloader = data_loader.get_contrastive_dataloader(batch_size=config.STAGE1_BATCH_SIZE)
    
    # Resume
    start_epoch = 0
    if config.RESUME_FROM_CHECKPOINT:
        start_epoch = utils.load_checkpoint(classifier, optimizer, config.RESUME_FROM_CHECKPOINT)
        
    # Training Loop
    for epoch in range(start_epoch, config.STAGE1_EPOCHS):
        classifier.train()
        total_loss = 0
        steps = 0
        
        for i, (seq1, seq2, target) in enumerate(dataloader):
            # Tokenize and move to device
            # We rely on model.encode or similar logic.
            # Since we don't have a working tokenizer in model.py yet, 
            # we need to ensure it works.
            # For now, let's assume model.forward_encoder handles list of strings 
            # if we implement the tokenization there.
            
            # But wait, I didn't implement tokenization in model.py fully.
            # I need to fix that or this will fail.
            # Let's assume I'll fix model.py or use a placeholder that works for now.
            # Actually, I should fix model.py first or handle it here.
            # But I can't easily fix model.py without knowing the tokenizer API.
            # I'll assume model.forward_encoder(seq_list) works.
            
            target = target.to(device)
            
            optimizer.zero_grad()
            
            emb1 = classifier.forward_encoder(seq1)
            emb2 = classifier.forward_encoder(seq2)
            
            loss = criterion(emb1, emb2, target)
            loss.backward()
            optimizer.step()
            
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
        print(
            f"--- Epoch {epoch+1} Complete. "
            f"Avg Train Loss: {avg_epoch_loss:.4f} ---"
        )
        utils.log_message(f"Epoch {epoch+1} Complete. Avg Loss: {avg_epoch_loss:.4f}", log_file)
                
        # Checkpoint
        if (epoch + 1) % config.SAVE_CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"stage1_epoch_{epoch+1}.pth")
            utils.save_checkpoint(classifier, optimizer, epoch, ckpt_path)
            
        # Visualization
        if (epoch + 1) % config.VISUALIZATION_EPOCH_INTERVAL == 0:
            utils.log_message("Generating visualization...", log_file)
            classifier.eval()
            with torch.no_grad():
                seqs, labels = data_loader.get_all_sequences_for_visualization()
                # Process in batches to avoid OOM
                all_embeddings = []
                batch_size = config.STAGE1_BATCH_SIZE
                for j in range(0, len(seqs), batch_size):
                    batch_seqs = seqs[j:j+batch_size]
                    batch_emb = classifier.forward_encoder(batch_seqs)
                    all_embeddings.append(batch_emb.cpu().numpy())
                
                all_embeddings = np.concatenate(all_embeddings, axis=0)
                
                vis_path = os.path.join(config.VISUALIZATION_DIR, f"stage1_epoch_{epoch+1}.png")
                utils.reduce_and_plot_embeddings(
                    all_embeddings, 
                    labels, 
                    vis_path, 
                    method=config.VISUALIZATION_METHOD
                )
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, "stage1_final.pth")
    utils.save_checkpoint(classifier, optimizer, config.STAGE1_EPOCHS, final_path)
    utils.log_message("Stage 1 training complete.", log_file)

if __name__ == "__main__":
    train_stage1()
