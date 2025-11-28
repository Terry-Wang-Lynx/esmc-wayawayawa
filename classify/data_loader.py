import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from . import config

def parse_fasta(fasta_path):
    """
    解析 FASTA 文件，返回 [(header, sequence), ...] 列表。
    """
    sequences = []
    if not os.path.exists(fasta_path):
        print(f"Warning: File not found: {fasta_path}")
        return sequences

    with open(fasta_path, 'r') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    sequences.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            sequences.append((header, "".join(seq)))
    return sequences

def augment_sequence(sequence):
    """
    对蛋白质序列应用随机突变。
    """
    if random.random() > config.AUGMENT_PROB:
        return sequence
        
    seq_list = list(sequence)
    # 标准氨基酸
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    for i in range(len(seq_list)):
        if random.random() < config.MUTATION_PROB:
            seq_list[i] = random.choice(amino_acids)
            
    return "".join(seq_list)

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels=None, augment=False):
        """
        蛋白质序列数据集。
        sequences: 序列字符串列表
        labels: 标签列表 (0 或 1)，可选
        augment: 是否应用数据增强
        """
        self.sequences = sequences
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        if self.augment:
            seq = augment_sequence(seq)
            
        if self.labels is not None:
            return seq, self.labels[idx]
        return seq

class ContrastiveDataset(Dataset):
    def __init__(self, mono_seqs, di_seqs):
        """
        对比学习数据集。
        mono_seqs: 类别 1 序列列表
        di_seqs: 类别 2 序列列表
        策略: 50% 正样本对 (同类), 50% 负样本对 (异类)
        """
        self.mono_seqs = mono_seqs
        self.di_seqs = di_seqs
        self.all_seqs = mono_seqs + di_seqs
        
    def __len__(self):
        return len(self.mono_seqs) + len(self.di_seqs)

    def __getitem__(self, idx):
        is_positive = random.random() > 0.5
        
        if is_positive:
            # 正样本对
            if random.random() > 0.5:
                seq1 = random.choice(self.mono_seqs)
                seq2 = random.choice(self.mono_seqs)
            else:
                seq1 = random.choice(self.di_seqs)
                seq2 = random.choice(self.di_seqs)
            target = 1.0  # 相似
        else:
            # 负样本对
            seq1 = random.choice(self.mono_seqs)
            seq2 = random.choice(self.di_seqs)
            target = -1.0  # 不相似
        
        return seq1, seq2, torch.tensor(target, dtype=torch.float)

def get_contrastive_dataloader(batch_size=config.STAGE1_BATCH_SIZE):
    train_mono = parse_fasta(os.path.join(config.DATA_ROOT, config.TRAIN_MONO_FILE))
    train_di = parse_fasta(os.path.join(config.DATA_ROOT, config.TRAIN_DI_FILE))
    
    mono_seqs = [s[1] for s in train_mono]
    di_seqs = [s[1] for s in train_di]
    
    dataset = ContrastiveDataset(mono_seqs, di_seqs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_classification_dataloader(batch_size=config.STAGE2_BATCH_SIZE, is_train=True):
    if is_train:
        mono_file = config.TRAIN_MONO_FILE
        di_file = config.TRAIN_DI_FILE
    else:
        mono_file = config.TEST_MONO_FILE
        di_file = config.TEST_DI_FILE
        
    mono_data = parse_fasta(os.path.join(config.DATA_ROOT, mono_file))
    di_data = parse_fasta(os.path.join(config.DATA_ROOT, di_file))
    
    sequences = [s[1] for s in mono_data] + [s[1] for s in di_data]
    # 标签: Mono=0, Di=1
    labels = [0] * len(mono_data) + [1] * len(di_data)
    
    # 仅训练时应用数据增强
    dataset = ProteinDataset(sequences, labels, augment=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_all_sequences_for_visualization():
    """
    返回测试集的 (sequences, labels, seq_ids) 用于可视化。
    """
    test_mono = parse_fasta(os.path.join(config.DATA_ROOT, config.TEST_MONO_FILE))
    test_di = parse_fasta(os.path.join(config.DATA_ROOT, config.TEST_DI_FILE))
    
    sequences = [s[1] for s in test_mono] + [s[1] for s in test_di]
    labels = [0] * len(test_mono) + [1] * len(test_di)
    seq_ids = [s[0] for s in test_mono] + [s[0] for s in test_di]
    return sequences, labels, seq_ids

def get_train_sequences_for_visualization(sample_size=2000):
    """
    返回训练集的 (sequences, labels, seq_ids) 用于可视化（采样）。
    """
    train_mono = parse_fasta(os.path.join(config.DATA_ROOT, config.TRAIN_MONO_FILE))
    train_di = parse_fasta(os.path.join(config.DATA_ROOT, config.TRAIN_DI_FILE))
    
    # 采样以避免数据过多
    if len(train_mono) > sample_size // 2:
        train_mono = random.sample(train_mono, sample_size // 2)
    if len(train_di) > sample_size // 2:
        train_di = random.sample(train_di, sample_size // 2)
        
    sequences = [s[1] for s in train_mono] + [s[1] for s in train_di]
    labels = [0] * len(train_mono) + [1] * len(train_di)
    seq_ids = [s[0] for s in train_mono] + [s[0] for s in train_di]
    return sequences, labels, seq_ids
