import os
import random
import torch
from torch.utils.data import Dataset

# --- 1. 来自 fasta.py 的逻辑 ---
def read_fasta(fasta_file):
    """
    读取 fasta 文件并返回一个 (name, sequence) 元组的列表。
    """
    fasta_dict = {}
    if not os.path.exists(fasta_file):
        print(f"[Warning] FASTA file not found: {fasta_file}")
        return []
        
    with open(fasta_file, 'r') as f:
        seq_name = None
        for line in f:
            if line.startswith('>'):
                seq_name = line.strip()[1:]
                fasta_dict[seq_name] = ""
            elif seq_name is not None:
                # 累加序列行，去除可能的换行符
                fasta_dict[seq_name] += line.strip()
    
    fasta_list = [(seq_name, seq) for seq_name, seq in fasta_dict.items()]
    return fasta_list

# --- 2. 来自 fasta_dataset.py 的逻辑 ---
class FastaDataset(Dataset):
    """
    基础 Fasta 数据集。
    - 如果提供了 label， __getitem__ 返回 ((name, seq), label)
    - 如果 label 为 None，__getitem__ 返回 (name, seq)
    """
    def __init__(self, fasta_file, label=None):
        super(FastaDataset, self).__init__()
        self.fasta_file = fasta_file
        self.label = label
        
        print(f"[DataLoader] Loading sequences from: {fasta_file}")
        self.fasta_list = read_fasta(fasta_file)
        print(f"             ...found {len(self.fasta_list)} sequences.")
    
    def __getitem__(self, index):
        if self.label is None:
            # 用于预测：返回 (name, seq)
            return self.fasta_list[index]
        else:
            # 用于训练：返回 ((name, seq), label)
            return self.fasta_list[index], self.label
    
    def __len__(self):
        return len(self.fasta_list)

# --- 3. 来自 training_dataset.py 的逻辑 ---
class TrainingDataset(Dataset):
    """
    混合正负样本，用于训练。
    """
    def __init__(self, positive_path, negative_path, dynamic_negative_sampling=False):
        super(TrainingDataset, self).__init__()
        self.dynamic_negative_sampling = dynamic_negative_sampling
        
        self.positive_dataset = FastaDataset(positive_path, label=1)
        self.negative_dataset = FastaDataset(negative_path, label=0)
        
        if len(self.positive_dataset) == 0 and len(self.negative_dataset) == 0:
            raise ValueError("Data error: Both positive and negative datasets are empty.")

    def __len__(self):
        if self.dynamic_negative_sampling:
            # 动态采样时，长度通常设置为正样本的 N 倍
            return len(self.positive_dataset) * 2
        else:
            return len(self.positive_dataset) + len(self.negative_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_dataset):
            return self.positive_dataset[idx]
        else:
            if self.dynamic_negative_sampling and len(self.negative_dataset) > 0:
                # 动态（随机）选择一个负样本
                return random.choice(self.negative_dataset)
            else:
                # 顺序选择负样本
                neg_idx = idx - len(self.positive_dataset)
                if len(self.negative_dataset) > 0:
                    neg_idx = neg_idx % len(self.negative_dataset) # 循环
                    return self.negative_dataset[neg_idx]
                else:
                    # 如果没有负样本（不推荐），则用正样本代替
                    return self.positive_dataset[idx % len(self.positive_dataset)]

# --- 4. 训练专用的 Collate 函数 ---
class TrainCollate:
    """
    在 DataLoader 中动态处理训练批次。
    输入: [ ((name, seq), label), ... ]
    输出: (padded_tokens_tensor, labels_tensor)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        
    def __call__(self, batch):
        # 1. 解包
        # data_tuples = [ (name, seq), ... ]
        # labels = [ 1, 0, 1, ... ]
        data_tuples, labels = zip(*batch)
        
        # 2. 提取序列字符串
        sequences = [item[1] for item in data_tuples]
        
        # 3. 动态 Tokenize
        token_list = [
            torch.tensor(self.tokenizer.encode(s, add_special_tokens=True)) 
            for s in sequences
        ]
        
        # 4. Padding
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            token_list, 
            batch_first=True, 
            padding_value=self.pad_id
        )
        
        # 5. 转换标签
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return padded_tokens, labels_tensor

# --- 5. 预测专用的 Collate 函数 ---
class PredictCollate:
    """
    在 DataLoader 中动态处理预测批次。
    输入: [ (name, seq), ... ]
    输出: (names_list, padded_tokens_tensor)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        
    def __call__(self, batch):
        # 1. 解包
        # names = [ name1, name2, ... ]
        # sequences = [ seq1, seq2, ... ]
        names, sequences = zip(*batch)
        
        # 2. 动态 Tokenize
        token_list = [
            torch.tensor(self.tokenizer.encode(s, add_special_tokens=True)) 
            for s in sequences
        ]
        
        # 3. Padding
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            token_list, 
            batch_first=True, 
            padding_value=self.pad_id
        )
        
        # 4. 返回序列名列表和 tokens 张量
        return list(names), padded_tokens