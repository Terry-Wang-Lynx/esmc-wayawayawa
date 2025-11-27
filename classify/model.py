import torch
import torch.nn as nn
import sys
import os

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrained import load_local_model, ESMC_600M

class ESMCClassifier(nn.Module):
    def __init__(self, embedding_dim=1152, freeze_base=True, unfreeze_last_n=0):
        super().__init__()
        # Load ESM-C 600M
        # We assume the weights are available as per pretrained.py logic
        print("Loading ESM-C model...")
        self.esmc = load_local_model(ESMC_600M, device="cpu") # Load on CPU first, move later
        print("ESM-C model loaded.")
        
        # Classification Head
        # Simple 2-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512), # Add normalization for stability
            nn.ReLU(),
            # nn.Dropout(0.1), # Remove dropout to allow overfitting on small data
            nn.Linear(512, 2) # 2 classes
        )
        
        self.freeze_base = freeze_base
        self.unfreeze_last_n = unfreeze_last_n
        self._setup_freezing()
        self.unfreeze_layers()
        
        # Ensure classifier head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    def _setup_freezing(self):
        if self.freeze_base:
            for param in self.esmc.parameters():
                param.requires_grad = False
            
            if self.unfreeze_last_n > 0:
                # Unfreeze last N layers
                # We need to know the structure of ESMC. 
                # Usually it has 'layers' or 'blocks'.
                # Let's try to find the transformer layers.
                # Based on pretrained.py: n_layers=36 for 600M.
                # Assuming structure is something like model.transformer.layers...
                # We will inspect named_parameters to be safe or just iterate children.
                # For now, let's try a heuristic: unfreeze parameters that contain "layer.XX" 
                # where XX is in the last N indices.
                
                # A more robust way without knowing exact structure:
                # Just unfreeze the last few parameters? No, that's risky.
                # Let's assume standard transformer structure.
                # If we can't find layers, we print a warning.
                
                # Let's try to find the main transformer module.
                # If it's ESMC, it might be self.esmc.transformer
                pass 

    def unfreeze_layers(self):
        # Actual implementation of unfreezing
        if not self.freeze_base:
            return

        if self.unfreeze_last_n > 0:
            # Get all parameter names
            param_names = [n for n, _ in self.esmc.named_parameters()]
            # Filter for layer indications. usually "layers.35." etc.
            # Let's assume the format involves "layers.{index}"
            
            # Find max layer index
            max_layer = -1
            import re
            # Updated regex to match "transformer.blocks.{index}."
            layer_pattern = re.compile(r"transformer\.blocks\.(\d+)\.")
            
            for name in param_names:
                match = layer_pattern.search(name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx > max_layer:
                        max_layer = layer_idx
            
            print(f"Max layer index found: {max_layer}")
            
            if max_layer != -1:
                target_start = max_layer - self.unfreeze_last_n + 1
                print(f"Unfreezing layers from index {target_start} to {max_layer}")
                count = 0
                for name, param in self.esmc.named_parameters():
                    match = layer_pattern.search(name)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx >= target_start:
                            param.requires_grad = True
                            count += 1
                            # print(f"Unfrozen {name}")
                print(f"Total parameters unfrozen: {count}")
            else:
                print("Warning: Could not detect layer structure to unfreeze specific layers.")
                # Fallback: Unfreeze everything if structure detection fails? 
                # Or maybe unfreeze the last few parameters blindly?
                # Let's print available parameter names to debug
                print("Available parameter names (first 20):")
                for n in param_names[:20]:
                    print(n)

    def forward_encoder(self, sequences):
        """
        sequences: list of protein sequence strings
        Returns: (B, D) embeddings
        """
        device = next(self.parameters()).device
        
        # Tokenize sequences using ESMC's internal tokenizer
        # ESMC has a _tokenize method that takes a list of sequences
        sequence_tokens = self.esmc._tokenize(sequences)  # Returns (B, L) tensor
        sequence_tokens = sequence_tokens.to(device)
        
        # Forward through ESMC
        output = self.esmc(sequence_tokens=sequence_tokens)
        
        # Extract embeddings
        # output.embeddings is (B, L, D)
        embeddings = output.embeddings
        
        # Mean pooling over sequence length
        # (B, L, D) -> (B, D)
        pooled = embeddings.mean(dim=1)
        
        # Safety check: Ensure output requires grad if we are training
        if self.training and not pooled.requires_grad:
             pooled.requires_grad_(True)
             
        return pooled

    def forward(self, tokens):
        embedding = self.forward_encoder(tokens)
        return self.classifier(embedding)

    def encode(self, sequence_list):
        # Helper to tokenize and encode
        # We need the tokenizer.
        # self.esmc.tokenizer
        
        # This is tricky without seeing ESMC code.
        # I'll assume I can use the tokenizer attached to the model.
        
        device = next(self.parameters()).device
        
        # Tokenization logic
        # If self.esmc has a tokenizer
        if hasattr(self.esmc, 'tokenizer'):
            # This is likely an EsmTokenizer object
            # It might have batch_encode_plus
            encoded = self.esmc.tokenizer.batch_encode_plus(sequence_list, padding=True, return_tensors="pt")
            input_ids = encoded['input_ids'].to(device)
            # attention_mask = encoded['attention_mask'].to(device)
            
            # Pass to model
            # We might need to pass attention_mask too
            # But forward_encoder above didn't take it.
            # Let's update forward_encoder to take input_ids and mask
            
            # For now, let's just return the embeddings from this method directly
            # to avoid signature mismatch in forward_encoder
            
            output = self.esmc(input_ids) # Maybe accepts mask?
            
            # Handle output...
            if hasattr(output, 'embeddings'):
                embeddings = output.embeddings
            else:
                embeddings = output # Assume tensor
            
            # Mean pool with mask
            # mask is (B, L)
            # embeddings is (B, L, D)
            # mask expanded: (B, L, 1)
            # sum(emb * mask) / sum(mask)
            
            # But wait, I don't know if batch_encode_plus exists.
            pass
            
        return None # Placeholder
