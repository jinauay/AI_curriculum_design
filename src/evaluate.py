import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import RumorDataset
from model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_df = pd.read_csv("data/val.csv")
val_dataset = RumorDataset(val_df['text'], val_df['label'])
val_loader = DataLoader(val_dataset, batch_size=16)

model = get_model()
model.load_state_dict(torch.load("checkpoints/best_model/bertweet.pth"))
model.to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"Validation Accuracy: {acc:.4f}")