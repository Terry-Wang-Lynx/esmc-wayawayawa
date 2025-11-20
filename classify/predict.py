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
    
    device = config.DEVICE
    print(f"[Predict] Using device: {device}")
    
    # Load Model
    classifier = model.ESMCClassifier(
        embedding_dim=config.EMBEDDING_DIM,
        freeze_base=True # Inference only
    ).to(device)
    
    model_path = os.path.join(config.CHECKPOINT_DIR, "stage2_final.pth")
    if not os.path.exists(model_path):
        print(f"[Predict] Error: Model not found at {model_path}")
        return
        
    print(f"[Predict] Loading fine-tuned weights from {model_path}...")
    utils.load_checkpoint(classifier, None, model_path)
    classifier.eval()
    
    # Load Data
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
    
    # Setup output directory
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    output_file = os.path.join(output_dir, "predictions_output.csv")
    class1_file = os.path.join(output_dir, "predictions_output_class1.csv")
    class1_fasta_file = os.path.join(output_dir, "predictions_output_class1.fasta")
    
    # Predict
    print("[Predict] Running inference...")
    results = []
    class1_results = []
    embeddings_list = []
    
    batch_size = config.STAGE2_BATCH_SIZE
    total_seqs = len(seq_list)
    processed_seqs = 0
    
    # Open output files
    with open(output_file, 'w') as f, \
         open(class1_file, 'w') as class1_f, \
         open(class1_fasta_file, 'w') as class1_fasta_f:
        
        # Write headers
        f.write("sequence_name,predicted_class,prob_class_0,prob_class_1\\n")
        class1_f.write("prob_class_1,sequence,predicted_class,sequence_name,prob_class_0\\n")
        
        with torch.no_grad():
            for i in range(0, len(seq_list), batch_size):
                batch_seqs = seq_list[i:i+batch_size]
                batch_headers = headers[i:i+batch_size]
                
                # Get embeddings and logits
                batch_emb = classifier.forward_encoder(batch_seqs)
                outputs = classifier.classifier(batch_emb)
                
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Store embeddings for visualization
                embeddings_list.append(batch_emb.cpu().numpy())
                
                # Process results
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
                    
                    # Write to main CSV
                    f.write(
                        f"{res['name']},{res['predicted_class']},"
                        f"{res['prob_class_0']:.6f},{res['prob_class_1']:.6f}\\n"
                    )
                    
                    # If predicted as class 1, write to class1 files
                    if res['predicted_class'] == 1:
                        class1_results.append(res)
                        
                        # Class 1 CSV
                        class1_f.write(
                            f"{res['prob_class_1']:.6f},"
                            f"{res['sequence']},"
                            f"{res['predicted_class']},"
                            f"{res['name']},"
                            f"{res['prob_class_0']:.6f}\\n"
                        )
                        
                        # Class 1 FASTA
                        class1_fasta_f.write(
                            f">{res['name']}\\n{res['sequence']}\\n"
                        )
                
                processed_seqs += len(batch_seqs)
                progress_ratio = processed_seqs / total_seqs if total_seqs > 0 else 0
                print(f"[Predict] Progress: {processed_seqs}/{total_seqs} ({progress_ratio*100:.2f}%)", end="\\r", flush=True)
                
                # Flush files
                f.flush()
                os.fsync(f.fileno())
                class1_f.flush()
                os.fsync(class1_f.fileno())
                class1_fasta_f.flush()
                os.fsync(class1_fasta_f.fileno())
    
    print()  # New line after progress
    
    # Display results
    print("\\n--- Prediction Results ---")
    print(f"Total sequences predicted: {len(results)}")
    
    # Print first 20 results
    for res in results[:20]:
        print(f"\\n> {res['name']}")
        print(f"  Predicted Class: {res['predicted_class']}")
        print(f"  Probabilities:   [Class 0: {res['prob_class_0']:.4f}, Class 1: {res['prob_class_1']:.4f}]")
    
    print(f"\\n[Predict] Class 1 count: {len(class1_results)}")
    for res in class1_results[:20]:
        print(f"\\n> {res['name']}")
        print(f"  Predicted Class: {res['predicted_class']}")
        print(f"  Probabilities:   [Class 0: {res['prob_class_0']:.4f}, Class 1: {res['prob_class_1']:.4f}]")
    
    print(f"\\n[Predict] Full results saved to {output_file}")
    print(f"[Predict] Class 1 CSV (with sequences) saved to {class1_file}")
    print(f"[Predict] Class 1 FASTA saved to {class1_fasta_file}")
    
    # Visualization
    print("\\n[Predict] Generating embedding visualization...")
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_predictions = [r['predicted_class'] for r in results]
    
    vis_path = os.path.join(config.VISUALIZATION_DIR, "prediction_vis.png")
    utils.reduce_and_plot_embeddings(
        all_embeddings, 
        all_predictions, 
        vis_path, 
        method=config.VISUALIZATION_METHOD
    )
    
    utils.log_message(f"Prediction complete. {len(results)} sequences processed.", log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict protein classes using ESM-C")
    parser.add_argument("input_fasta", help="Path to input FASTA file")
    parser.add_argument("--output_dir", help="Path to output directory", default=None)
    
    args = parser.parse_args()
    predict(args.input_fasta, args.output_dir)
