import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from . import config

def parse_fasta(fasta_path):
    """
    Parses a FASTA file and returns a list of (header, sequence) tuples.
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
    Applies random mutation to a protein sequence string.
    """
    if random.random() > config.AUGMENT_PROB:
        return sequence
        
    seq_list = list(sequence)
    # Standard Amino Acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    for i in range(len(seq_list)):
        if random.random() < config.MUTATION_PROB:
            seq_list[i] = random.choice(amino_acids)
            
    return "".join(seq_list)

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels=None, augment=False):
        """
        sequences: list of strings (protein sequences)
        labels: list of int (0 or 1), optional
        augment: bool, whether to apply data augmentation
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
        mono_seqs: list of sequences from class 1
        di_seqs: list of sequences from class 2
        """
        self.mono_seqs = mono_seqs
        self.di_seqs = di_seqs
        self.all_seqs = mono_seqs + di_seqs
        # Create a map for faster positive sampling if needed, 
        # but for contrastive learning we usually just need pairs.
        # Strategy:
        # 50% chance: Positive pair (same class)
        # 50% chance: Negative pair (different class)
        
    def __len__(self):
        # Arbitrary length, usually sum of both
        return len(self.mono_seqs) + len(self.di_seqs)

    def __getitem__(self, idx):
        # Randomly select a strategy
        is_positive = random.random() > 0.5
        
        if is_positive:
            # Choose a class
            if random.random() > 0.5:
                # Mono class
                seq1 = random.choice(self.mono_seqs)
                seq2 = random.choice(self.mono_seqs)
            else:
                # Di class
                seq1 = random.choice(self.di_seqs)
                seq2 = random.choice(self.di_seqs)
            target = 1.0 # Similar
        else:
            # Negative pair
            seq1 = random.choice(self.mono_seqs)
            seq2 = random.choice(self.di_seqs)
            target = -1.0 # Dissimilar
        
        # Note: We could also apply augmentation here if we wanted to make contrastive learning harder
        # But for now let's keep it simple as requested for classification task
            
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
    # Label 0 for Mono, 1 for Di (or vice versa, let's stick to 0 and 1)
    # Config doesn't specify which is which, let's assume Mono=0, Di=1
    labels = [0] * len(mono_data) + [1] * len(di_data)
    
    # Apply augmentation only if training
    dataset = ProteinDataset(sequences, labels, augment=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_all_sequences_for_visualization():
    """
    Returns (sequences, labels, seq_ids) for test set visualization
    """
    test_mono = parse_fasta(os.path.join(config.DATA_ROOT, config.TEST_MONO_FILE))
    test_di = parse_fasta(os.path.join(config.DATA_ROOT, config.TEST_DI_FILE))
    
    sequences = [s[1] for s in test_mono] + [s[1] for s in test_di]
    labels = [0] * len(test_mono) + [1] * len(test_di)
    seq_ids = [s[0] for s in test_mono] + [s[0] for s in test_di]
    return sequences, labels, seq_ids

def get_train_sequences_for_visualization(sample_size=2000):
    """
    Returns (sequences, labels, seq_ids) for training set visualization (sampled)
    """
    train_mono = parse_fasta(os.path.join(config.DATA_ROOT, config.TRAIN_MONO_FILE))
    train_di = parse_fasta(os.path.join(config.DATA_ROOT, config.TRAIN_DI_FILE))
    
    # Sample if too large
    if len(train_mono) > sample_size // 2:
        train_mono = random.sample(train_mono, sample_size // 2)
    if len(train_di) > sample_size // 2:
        train_di = random.sample(train_di, sample_size // 2)
        
    sequences = [s[1] for s in train_mono] + [s[1] for s in train_di]
    labels = [0] * len(train_mono) + [1] * len(train_di)
    seq_ids = [s[0] for s in train_mono] + [s[0] for s in train_di]
    return sequences, labels, seq_ids
