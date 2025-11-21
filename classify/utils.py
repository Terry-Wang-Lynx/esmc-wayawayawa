import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from . import config

def log_message(message, log_file):
    """
    Appends a message with timestamp to the log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    
    with open(log_file, 'a') as f:
        f.write(full_message + "\n")

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Saves model and optimizer state.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Loads model and optimizer state. Returns the next epoch.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found: {filepath}")
        return 0
        
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch'] + 1

def reduce_and_plot_embeddings(embeddings, labels, output_path, method='UMAP'):
    """
    Reduces embeddings to 2D and plots them.
    embeddings: numpy array (N, D)
    labels: list or array of labels
    """
    try:
        if method == 'UMAP':
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto')
            
        embedding_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        # Assuming binary labels 0 and 1
        unique_labels = np.unique(labels)
        colors = ['blue', 'red']
        class_names = ['Mono', 'Di']
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=colors[i % len(colors)],
                label=class_names[int(label)] if int(label) < 2 else f"Class {label}",
                alpha=0.6,
                s=10
            )
            
        plt.title(f"Protein Embeddings Visualization ({method})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to temporary file first
        temp_path = output_path + ".tmp.png"
        plt.savefig(temp_path)
        plt.close()
        
        # Rename to target
        os.rename(temp_path, output_path)
        print(f"Visualization saved to {output_path}")
        
    except ImportError as e:
        print(f"Visualization failed: Missing dependency {e}")
    except Exception as e:
        print(f"Visualization failed: {e}")

def save_training_plots(history, output_dir):
    """
    Plots training history (Loss, etc.)
    history: dict with keys like 'train_loss', 'epoch'
    """
    epochs = history.get('epoch', [])
    if not epochs:
        epochs = list(range(1, len(history.get('train_loss', [])) + 1))
        
    # Plot Loss
    if 'train_loss' in history:
        plt.figure()
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        if 'val_loss' in history and len(history['val_loss']) == len(epochs):
            plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        plt.close()
        
    # Plot Accuracy if available
    if 'train_accuracy' in history:
        plt.figure()
        plt.plot(epochs, history['train_accuracy'], label='Train Acc')
        if 'val_accuracy' in history and len(history['val_accuracy']) == len(epochs):
            plt.plot(epochs, history['val_accuracy'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_accuracy.png'))
        plt.close()

def reduce_and_plot_train_test_embeddings(train_emb, train_lbl, test_emb, test_lbl, output_path, method='UMAP'):
    """
    Plots both train and test embeddings on the same plot.
    """
    try:
        # Combine for reduction to ensure same space
        all_emb = np.concatenate([train_emb, test_emb], axis=0)
        
        if method == 'UMAP':
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto')
            
        all_emb_2d = reducer.fit_transform(all_emb)
        
        # Split back
        n_train = len(train_emb)
        train_2d = all_emb_2d[:n_train]
        test_2d = all_emb_2d[n_train:]
        
        plt.figure(figsize=(12, 10))
        
        # Plot Train (lighter, smaller)
        unique_labels = np.unique(train_lbl)
        colors = ['lightblue', 'lightcoral'] # Lighter versions of blue/red
        class_names = ['Train Mono', 'Train Di']
        
        for i, label in enumerate(unique_labels):
            mask = np.array(train_lbl) == label
            plt.scatter(
                train_2d[mask, 0],
                train_2d[mask, 1],
                c=colors[i % len(colors)],
                label=class_names[int(label)],
                alpha=0.3,
                s=10,
                marker='o'
            )
            
        # Plot Test (darker, larger or different marker)
        unique_labels = np.unique(test_lbl)
        colors = ['blue', 'red']
        class_names = ['Test Mono', 'Test Di']
        
        for i, label in enumerate(unique_labels):
            mask = np.array(test_lbl) == label
            plt.scatter(
                test_2d[mask, 0],
                test_2d[mask, 1],
                c=colors[i % len(colors)],
                label=class_names[int(label)],
                alpha=0.8,
                s=20,
                marker='x'
            )
            
        plt.title(f"Protein Embeddings Visualization ({method})\nTrain (Circle) vs Test (Cross)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
