import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from dataset import RumorDataset
from model import get_model
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

train_dataset = RumorDataset(train_df['text'], train_df['label'])
val_dataset = RumorDataset(val_df['text'], val_df['label'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 模型
model = get_model()
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练循环
best_acc = 0.0
for epoch in range(3):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 验证
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
    print(f"Epoch {epoch+1}: Val Accuracy = {acc:.4f}")

    # 保存最优模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "checkpoints/best_model/bertweet.pth")