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
    log_file = os.path.join(config.LOG_DIR, "stage2_log.txt")
    utils.log_message("Starting Stage 2: Classification Finetuning", log_file)
    
    device = config.DEVICE
    utils.log_message(f"Using device: {device}", log_file)
    
    # Model
    # Initialize model
    classifier = model.ESMCClassifier(
        embedding_dim=config.EMBEDDING_DIM,
        freeze_base=config.FREEZE_BASE_MODEL,
        unfreeze_last_n=config.UNFREEZE_LAST_N_LAYERS
    ).to(device)
    
    # Load Stage 1 weights if available
    stage1_weights = os.path.join(config.CHECKPOINT_DIR, "stage1_final.pth")
    if os.path.exists(stage1_weights):
        utils.log_message(f"Loading Stage 1 weights from {stage1_weights}", log_file)
        # We only want to load the encoder part, not the optimizer or the whole state if keys mismatch.
        # But wait, Stage 1 saved the whole model state dict.
        # Stage 1 model had the same architecture (ESMCClassifier) but the head might be random or unused?
        # Actually Stage 1 used ESMCClassifier but only trained the encoder (via forward_encoder).
        # The classifier head in Stage 1 was initialized but not trained.
        # So we can load the encoder weights.
        
        checkpoint = torch.load(stage1_weights, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Filter out classifier keys if we want to start fresh head, 
        # but actually we want to keep the encoder.
        # The keys are likely "esmc..." and "classifier..."
        # We should load "esmc..." keys.
        
        model_dict = classifier.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and k.startswith('esmc')}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        classifier.load_state_dict(model_dict)
        
        utils.log_message("Stage 1 encoder weights loaded.", log_file)
    else:
        utils.log_message("Warning: Stage 1 weights not found. Training from scratch/pretrained base.", log_file)
    
    # Apply freezing logic again just in case loading state dict messed it up (unlikely but good practice)
    classifier._setup_freezing()
    classifier.unfreeze_layers() # Apply unfreezing logic
    
    # Optimizer
    # We only optimize parameters that require grad
    params_to_optimize = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=config.STAGE2_LEARNING_RATE)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Data
    train_loader = data_loader.get_classification_dataloader(batch_size=config.STAGE2_BATCH_SIZE, is_train=True)
    test_loader = data_loader.get_classification_dataloader(batch_size=config.STAGE2_BATCH_SIZE, is_train=False)
    
    # Training Loop
    for epoch in range(config.STAGE2_EPOCHS):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (seqs, labels) in enumerate(train_loader):
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = classifier(seqs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Real-time step output (every 10 steps)
            if (i + 1) % 10 == 0:
                batch_acc = (predicted == labels).float().mean().item() if labels.size(0) > 0 else 0.0
                avg_loss = total_loss / (i + 1)
                print(f"  Epoch {epoch+1}/{config.STAGE2_EPOCHS}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Batch Acc: {batch_acc * 100:.2f}%")
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"--- Epoch {epoch+1} Complete. Avg Train Loss: {avg_loss:.4f}, Avg Train Acc: {accuracy:.2f}% ---")
        utils.log_message(f"Epoch [{epoch+1}/{config.STAGE2_EPOCHS}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%", log_file)
        
        # Evaluation
        if (epoch + 1) % config.EVAL_EPOCH_INTERVAL == 0:
            classifier.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for seqs, labels in test_loader:
                    labels = labels.to(device)
                    outputs = classifier(seqs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = 100 * test_correct / test_total
            utils.log_message(f"Test Accuracy: {test_acc:.2f}%", log_file)
            
        # Visualization
        if (epoch + 1) % config.VISUALIZATION_EPOCH_INTERVAL == 0:
            utils.log_message("Generating visualization...", log_file)
            classifier.eval()
            with torch.no_grad():
                seqs, labels = data_loader.get_all_sequences_for_visualization()
                all_embeddings = []
                batch_size = config.STAGE2_BATCH_SIZE
                for j in range(0, len(seqs), batch_size):
                    batch_seqs = seqs[j:j+batch_size]
                    batch_emb = classifier.forward_encoder(batch_seqs)
                    all_embeddings.append(batch_emb.cpu().numpy())
                
                all_embeddings = np.concatenate(all_embeddings, axis=0)
                
                vis_path = os.path.join(config.VISUALIZATION_DIR, f"stage2_epoch_{epoch+1}.png")
                utils.reduce_and_plot_embeddings(
                    all_embeddings, 
                    labels, 
                    vis_path, 
                    method=config.VISUALIZATION_METHOD
                )
                
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, "stage2_final.pth")
    utils.save_checkpoint(classifier, optimizer, config.STAGE2_EPOCHS, final_path)
    utils.log_message("Stage 2 training complete.", log_file)

if __name__ == "__main__":
    train_stage2()
