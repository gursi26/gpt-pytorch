from torch.utils.data import Dataset
import pandas as pd
import torch

class QQPDataset(Dataset):

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.drop(["id", "qid1", "qid2"], axis=1)
        self.df.columns = ["q1", "q2", "label"]
        self.df = self.df.dropna()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        q1, q2, label = self.df.iloc[idx]
        order1 = f"<bos> {q1} <sep> {q2} <eos>"
        order2 = f"<bos> {q2} <sep> {q1} <eos>"
        return order1, order2, label
    

def prepare_mask(padding_mask, causal=True):
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(-2)
    if causal:
        causal = torch.tril(torch.ones(padding_mask.shape[-1], padding_mask.shape[-1])).type(torch.bool)
        padding_mask = padding_mask * causal.to(padding_mask.device)
    return padding_mask


class TokenizeCollate:

    def __init__(self, tokenizer_obj):
        self.tokenizer = tokenizer_obj

    def __call__(self, x):
        q1, q2, labels = [], [], []
        for q1_item, q2_item, label in x:
            q1.append(q1_item)
            q2.append(q2_item)
            labels.append(label)
        q1 = self.tokenizer(q1, return_tensors='pt', padding=True)
        q2 = self.tokenizer(q2, return_tensors='pt', padding=True)
        labels = torch.tensor(labels).type(torch.float32)
        return q1["input_ids"], q2["input_ids"], q1["attention_mask"].type(torch.bool), q2["attention_mask"].type(torch.bool), labels