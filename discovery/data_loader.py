import os
import random
import torch
from torch.utils.data import Dataset

# --- 全局最大序列长度 ---
# ESM 模型的上下文窗口通常是 1024。超过此长度的序列将被截断。
# 可根据显存情况调整（例如 512 / 2048）。
MAX_SEQUENCE_LENGTH = 1024


# --- 1. 读取 FASTA 的工具函数 ---
def read_fasta(fasta_file):
    """
    读取 fasta 文件并返回一个 [(name, sequence), ...] 的列表。

    注意：
    不能用 dict 以 name 作为 key 来保存序列，
    否则当存在重复的 header（name 完全相同）时，前面的会被后面的覆盖，
    这会导致条目数变少（你现在 156 条变成 72 条就是这个原因）。
    """
    if not os.path.exists(fasta_file):
        print(f"[Warning] FASTA file not found: {fasta_file}")
        return []

    fasta_list = []
    seq_name = None
    seq_chunks = []

    with open(fasta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # 遇到新的 header 时，如果前一个序列存在，则先保存
                if seq_name is not None:
                    seq = "".join(seq_chunks)
                    fasta_list.append((seq_name, seq))
                seq_name = line[1:].strip()
                seq_chunks = []
            else:
                # 累加序列行
                if seq_name is not None:
                    seq_chunks.append(line)

    # 文件结束时，把最后一条序列加入列表
    if seq_name is not None:
        seq = "".join(seq_chunks)
        fasta_list.append((seq_name, seq))

    print(f"[read_fasta] Loaded {len(fasta_list)} sequences from {fasta_file}")
    return fasta_list


# --- 2. 基础 Fasta 数据集 ---
class FastaDataset(Dataset):
    """
    基础 Fasta 数据集。
    - 如果提供了 label，__getitem__ 返回 ((name, seq), label)
    - 如果 label 为 None，__getitem__ 返回 (name, seq)
    """
    def __init__(self, fasta_file, label=None):
        super(FastaDataset, self).__init__()
        self.fasta_file = fasta_file
        self.label = label

        print(f"[DataLoader] Loading sequences from: {fasta_file}")
        self.fasta_list = read_fasta(fasta_file)
        print(f"[DataLoader]   ...found {len(self.fasta_list)} sequences.")

    def __getitem__(self, index):
        if self.label is None:
            # 用于预测：返回 (name, seq)
            return self.fasta_list[index]
        else:
            # 用于训练：返回 ((name, seq), label)
            return self.fasta_list[index], self.label

    def __len__(self):
        return len(self.fasta_list)


# --- 3. 训练用混合数据集（正 / 负样本 + 动态负采样） ---
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
            # 动态采样时，长度通常设置为正样本的 N 倍（这里取 2x，可自行调整）
            return len(self.positive_dataset) * 2
        else:
            return len(self.positive_dataset) + len(self.negative_dataset)

    def __getitem__(self, idx):
        # 先保证正样本都能被遍历到
        if idx < len(self.positive_dataset):
            return self.positive_dataset[idx]
        else:
            # 负样本部分
            if self.dynamic_negative_sampling and len(self.negative_dataset) > 0:
                # 动态负采样：每次随机选取一个负样本
                neg_idx = random.randint(0, len(self.negative_dataset) - 1)
                return self.negative_dataset[neg_idx]
            else:
                # 顺序选择负样本（或在没有动态采样时）
                neg_idx = idx - len(self.positive_dataset)
                if len(self.negative_dataset) > 0:
                    neg_idx = neg_idx % len(self.negative_dataset)  # 循环
                    return self.negative_dataset[neg_idx]
                else:
                    # 如果没有负样本（不推荐），则用正样本代替，防止崩溃
                    return self.positive_dataset[idx % len(self.positive_dataset)]


# --- 4. 训练专用 Collate（只返回 tokens 与 labels） ---
class TrainCollate:
    """
    在 DataLoader 中动态处理训练批次。
    输入:  [ ((name, seq), label), ... ]
    输出:  (padded_tokens_tensor, labels_tensor)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.max_length = MAX_SEQUENCE_LENGTH

    def __call__(self, batch):
        # 1. 解包
        # data_tuples = [ (name, seq), ... ]
        # labels = [ 1, 0, 1, ... ]
        data_tuples, labels = zip(*batch)

        # 2. 提取序列字符串和名字
        sequences = [item[1] for item in data_tuples]
        names = [item[0] for item in data_tuples]

        # 3. 动态 Tokenize + 截断
        token_list = [
            torch.tensor(
                self.tokenizer.encode(seq, add_special_tokens=True)[: self.max_length]
            )
            for seq in sequences
        ]

        # 4. Padding
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            token_list,
            batch_first=True,
            padding_value=self.pad_id,
        )

        # 5. 转换标签
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return padded_tokens, labels_tensor, list(names)


# --- 5. 预测专用 Collate（返回 name / tokens / 原始序列） ---
class PredictCollate:
    """
    在 DataLoader 中动态处理预测批次。
    输入:  [ (name, seq), ... ]
    输出:  (names_list, padded_tokens_tensor, sequences_list)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.max_length = MAX_SEQUENCE_LENGTH

    def __call__(self, batch):
        # 1. 解包
        # names = [ name1, name2, ... ]
        # sequences = [ seq1, seq2, ... ]
        names, sequences = zip(*batch)

        # 2. 动态 Tokenize + 截断
        token_list = [
            torch.tensor(
                self.tokenizer.encode(seq, add_special_tokens=True)[: self.max_length]
            )
            for seq in sequences
        ]

        # 3. Padding
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            token_list,
            batch_first=True,
            padding_value=self.pad_id,
        )

        # 4. 清洗序列（去掉所有空白，确保无空格 / 换行）
        clean_sequences = ["".join(str(s).split()) for s in sequences]

        # 5. 返回：名字、tokens、干净的氨基酸序列字符串
        return list(names), padded_tokens, clean_sequences