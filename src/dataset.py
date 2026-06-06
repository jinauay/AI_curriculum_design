import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from preprocess import clean_text

class RumorDataset(Dataset):
    def __init__(self, texts, labels, model_name="vinai/bertweet-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.texts = [clean_text(t) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item