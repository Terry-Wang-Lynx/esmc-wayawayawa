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
