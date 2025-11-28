import torch
import torch.nn as nn
from esm.models.esmc import ESMC, ESMCOutput
# 1. 导入 ESMC tokenizer 和 model config
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.models import ESMC_600M # 用于检查模型名称
import os # 用于检查文件
from esm.models.esmc import ESMC # 确保 ESMC 被导入

class ESMCForSequenceClassification(nn.Module):
    """
    一个包装 ESMC 模型的二分类头。
    它使用 [CLS] token 进行分类，并实现了部分层解冻的微调策略。
    """
    def __init__(self, esmc_model: ESMC, num_classes: int = 2, unfreeze_last_n_layers: int = 2):
        """
        Args:
            esmc_model: 预训练的 ESMC 模型实例
            num_classes: 分类类别数 (默认为 2)
            unfreeze_last_n_layers: 需要解冻参与训练的最后几层 Transformer 层数
        """
        super().__init__()
        self.esmc = esmc_model
        
        # 获取 d_model 维度 (对于 ESMC-600M 是 1152)
        self.hidden_dim = esmc_model.embed.embedding_dim
        
        # 定义简单的分类头
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
        # 执行冻结/解冻策略
        self._setup_gradients(unfreeze_last_n_layers)

    def _setup_gradients(self, n_layers: int):
        # 1. 首先冻结 ESMC 主干的所有参数
        for param in self.esmc.parameters():
            param.requires_grad = False
            
        # 2. 确保分类头是可训练的
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        # 3. 解冻 Transformer 的最后 n 层
        blocks = self.esmc.transformer.blocks
        if n_layers > 0 and len(blocks) > 0:
            # 切片获取最后 n 层
            layers_to_unfreeze = blocks[-n_layers:]
            print(f"[Model] Unfreezing last {len(layers_to_unfreeze)} transformer blocks...")
            for block in layers_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
                    
        # 4. 解冻 Transformer 最终的 LayerNorm 层
        if hasattr(self.esmc.transformer, 'norm'):
            print("[Model] Unfreezing final LayerNorm...")
            for param in self.esmc.transformer.norm.parameters():
                param.requires_grad = True

    def forward(self, sequence_tokens: torch.Tensor, sequence_id: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            sequence_tokens: [Batch, Length] 经过 tokenize 的输入
            sequence_id: [Batch, Length] Mask 矩阵 (可选)
        """
        # 1. 通过 ESMC 主干网络
        outputs: ESMCOutput = self.esmc(sequence_tokens, sequence_id=sequence_id)
        
        # 2. 提取 Embeddings
        embeddings = outputs.embeddings
        
        # 3. 提取 CLS Token (位于序列索引 0)
        cls_embedding = embeddings[:, 0, :]  # Shape: [Batch, Hidden_Dim]
        
        # 4. 通过分类头
        logits = self.classifier(cls_embedding)
        
        return logits

    def forward_encoder(self, sequence_tokens: torch.Tensor, sequence_id: torch.Tensor = None) -> torch.Tensor:
        """
        仅提取编码器的输出 (CLS token embedding)，用于可视化或特征提取。
        """
        # 1. 通过 ESMC 主干网络
        outputs: ESMCOutput = self.esmc(sequence_tokens, sequence_id=sequence_id)
        
        # 2. 提取 Embeddings
        embeddings = outputs.embeddings
        
        # 3. 提取 CLS Token (位于序列索引 0)
        cls_embedding = embeddings[:, 0, :]  # Shape: [Batch, Hidden_Dim]
        
        return cls_embedding

def get_model_and_tokenizer(
    model_name: str = "esmc_600m",
    num_classes: int = 2,
    unfreeze_layers: int = 2,
    device: torch.device = None,
    local_weights_path: str = None # ! 这是修复错误的关键
):
    """
    加载基础 ESMC 模型、Tokenizer，并返回包装后的分类模型。
    如果提供了 local_weights_path，将从本地文件加载权重，
    否则将尝试使用 ESMC.from_pretrained（这可能会触发下载）。
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"[Model] Using device: {device}")
    
    base_model: ESMC
    
    if local_weights_path:
        print(f"[Model] Loading ESMC base model '{model_name}' from local path: {local_weights_path}")
        
        if not os.path.exists(local_weights_path):
            print(f"[Model] Error: Local weights file not found at {local_weights_path}")
            raise FileNotFoundError(f"Weight file not found: {local_weights_path}")

        # --- 手动构建模型 ---
        # 1. 定义模型参数 (esmc_600m)
        if model_name != ESMC_600M:
            print(f"Warning: local_weights_path is provided, but model_name is '{model_name}'. Assuming weights match {ESMC_600M} architecture.")
        
        d_model = 1152
        n_heads = 18
        n_layers = 36
        tokenizer = get_esmc_model_tokenizers()
        
        # 2. 实例化模型
        # (我们必须在 CPU 上实例化，然后再加载，最后 .to(device))
        with torch.device("cpu"):
             base_model = ESMC(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                tokenizer=tokenizer,
                use_flash_attn=True, # 假设
            ).eval()
        
        # 3. 从本地路径加载权重
        try:
            state_dict = torch.load(local_weights_path, map_location="cpu") # 先加载到 CPU
            base_model.load_state_dict(state_dict)
            base_model = base_model.to(device) # 然后移动到目标 device
            print("[Model] Local weights loaded successfully.")
        except Exception as e:
            print(f"[Model] Error loading local weights: {e}")
            raise
            
    else:
        # 原始逻辑：从 HF Hub 下载
        print(f"[Model] Loading ESMC base model '{model_name}' using from_pretrained (will download if not cached)...")
        base_model = ESMC.from_pretrained(model_name, device=device)
        print("[Model] Base model loaded from_pretrained.")

    tokenizer = base_model.tokenizer
    
    # 4. 包装模型
    model = ESMCForSequenceClassification(
        base_model,
        num_classes=num_classes,
        unfreeze_last_n_layers=unfreeze_layers
    ).to(device)
    
    return model, tokenizer