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
    
    # Ensure the parent directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
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

def save_metrics_to_csv(metrics_dict, csv_path):
    """
    Saves metrics to a CSV file. Creates the file with headers if it doesn't exist,
    otherwise appends a new row.
    
    Args:
        metrics_dict: Dictionary with metric names as keys and values as floats/ints
        csv_path: Path to the CSV file
    """
    import csv
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Open file in append mode
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write metrics row
        writer.writerow(metrics_dict)

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
        
    # Plot Contrastive Similarities if available
    if 'pos_sim' in history and 'neg_sim' in history:
        plt.figure()
        plt.plot(epochs, history['pos_sim'], label='Avg Pos Sim', color='green')
        plt.plot(epochs, history['neg_sim'], label='Avg Neg Sim', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.title('Contrastive Learning Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'contrastive_metrics.png'))
        plt.close()
    
    # Plot Test Accuracy if available
    if 'test_acc' in history and len(history['test_acc']) > 0:
        plt.figure()
        test_epochs = [history['epoch'][i] for i in range(len(history['test_acc']))]
        plt.plot(test_epochs, history['test_acc'], label='Test Acc', marker='o')
        if 'train_acc' in history:
            plt.plot(epochs, history['train_acc'], label='Train Acc', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'))
        plt.close()
    
    # Plot Classification Metrics (Precision, Recall, F1) if available
    if 'test_precision' in history and len(history['test_precision']) > 0:
        plt.figure()
        test_epochs = [history['epoch'][i] for i in range(len(history['test_precision']))]
        plt.plot(test_epochs, history['test_precision'], label='Precision', marker='o')
        plt.plot(test_epochs, history['test_recall'], label='Recall', marker='s')
        plt.plot(test_epochs, history['test_f1'], label='F1 Score', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Classification Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'classification_metrics.png'))
        plt.close()

def reduce_and_plot_train_test_embeddings(train_emb, train_lbl, test_emb, test_lbl, output_path, method='UMAP', train_ids=None, test_ids=None):
    """
    Plots both train and test embeddings on the same plot.
    Also saves a CSV file with coordinates and sequence IDs.
    
    Args:
        train_emb: Training embeddings
        train_lbl: Training labels
        test_emb: Test embeddings
        test_lbl: Test labels
        output_path: Path to save the plot
        method: Dimensionality reduction method ('UMAP' or 't-SNE')
        train_ids: Optional list of training sequence IDs
        test_ids: Optional list of test sequence IDs
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
        
        # Save coordinates to CSV in separate directory
        # Extract filename from output_path
        import os
        plot_filename = os.path.basename(output_path)
        csv_filename = plot_filename.replace('.png', '_coordinates.csv')
        
        # Get the parent directory of the plots directory
        # output_path is like: .../visualizations/plots/stage1_epoch_10.png
        # We want coords_dir to be: .../visualizations/coordinates/
        output_dir = os.path.dirname(output_path)  # .../visualizations/plots
        parent_dir = os.path.dirname(output_dir)    # .../visualizations
        coords_dir = os.path.join(parent_dir, 'coordinates')
        
        os.makedirs(coords_dir, exist_ok=True)
        csv_path = os.path.join(coords_dir, csv_filename)
        
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sequence_id', 'dataset', 'label', 'x', 'y'])
            
            # Write train data
            for i in range(len(train_2d)):
                seq_id = train_ids[i] if train_ids else f'train_{i}'
                label_name = 'mono' if train_lbl[i] == 0 else 'di'
                writer.writerow([seq_id, 'train', label_name, train_2d[i, 0], train_2d[i, 1]])
            
            # Write test data
            for i in range(len(test_2d)):
                seq_id = test_ids[i] if test_ids else f'test_{i}'
                label_name = 'mono' if test_lbl[i] == 0 else 'di'
                writer.writerow([seq_id, 'test', label_name, test_2d[i, 0], test_2d[i, 1]])
        
        print(f"Coordinates saved to {csv_path}")
        
        # Plot
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

def plot_confusion_matrix(cm, classes, output_path, title='Confusion Matrix'):
    """
    Plots the confusion matrix using matplotlib.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_scores, output_path):
    """
    Plots the ROC curve.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, output_path):
    """
    Plots the Precision-Recall curve.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
