import torch
import torch.nn as nn
import sys
import os

# 确保可以从根目录导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrained import load_local_model, ESMC_600M

class ESMCClassifier(nn.Module):
    def __init__(self, embedding_dim=1152, freeze_base=True, unfreeze_last_n=0):
        super().__init__()
        # 加载 ESM-C 600M
        print("Loading ESM-C model...")
        self.esmc = load_local_model(ESMC_600M, device="cpu")
        print("ESM-C model loaded.")
        
        # 分类头: 简单的 2 层 MLP
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 2)  # 2 分类
        )
        
        self.freeze_base = freeze_base
        self.unfreeze_last_n = unfreeze_last_n
        self._setup_freezing()
        self.unfreeze_layers()
        
        # 确保分类头可训练
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    def _setup_freezing(self):
        if self.freeze_base:
            for param in self.esmc.parameters():
                param.requires_grad = False

    def unfreeze_layers(self):
        """解冻 ESMC 最后 N 层"""
        if not self.freeze_base:
            return

        if self.unfreeze_last_n > 0:
            param_names = [n for n, _ in self.esmc.named_parameters()]
            
            # 查找最大层索引
            max_layer = -1
            import re
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
                print(f"Total parameters unfrozen: {count}")
            else:
                print("Warning: Could not detect layer structure to unfreeze specific layers.")
                print("Available parameter names (first 20):")
                for n in param_names[:20]:
                    print(n)

    def forward_encoder(self, sequences):
        """
        提取序列的 Embedding。
        sequences: 蛋白质序列字符串列表
        返回: (B, D) embeddings
        """
        device = next(self.parameters()).device
        
        # 使用 ESMC 内部 tokenizer
        sequence_tokens = self.esmc._tokenize(sequences)
        sequence_tokens = sequence_tokens.to(device)
        
        # 前向传播
        output = self.esmc(sequence_tokens=sequence_tokens)
        
        # 提取 embeddings (B, L, D)
        embeddings = output.embeddings
        
        # 平均池化 (B, L, D) -> (B, D)
        pooled = embeddings.mean(dim=1)
        
        # 确保训练时需要梯度
        if self.training and not pooled.requires_grad:
             pooled.requires_grad_(True)
             
        return pooled

    def forward(self, tokens):
        embedding = self.forward_encoder(tokens)
        return self.classifier(embedding)

    def encode(self, sequence_list):
        """
        辅助函数: tokenize 并编码序列。
        (预留接口，当前未完全实现)
        """
        device = next(self.parameters()).device
        
        if hasattr(self.esmc, 'tokenizer'):
            encoded = self.esmc.tokenizer.batch_encode_plus(sequence_list, padding=True, return_tensors="pt")
            input_ids = encoded['input_ids'].to(device)
            
            output = self.esmc(input_ids)
            
            if hasattr(output, 'embeddings'):
                embeddings = output.embeddings
            else:
                embeddings = output
            
        return None  # 占位符
