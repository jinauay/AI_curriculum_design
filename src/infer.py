import torch
from transformers import AutoTokenizer
from model import get_model
from explain import get_explanation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model()
model.load_state_dict(torch.load("checkpoints/best_model/bertweet.pth"))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, label].item()
    return label, confidence

if __name__ == "__main__":
    text = input("请输入推文: ")
    label, confidence = predict(text)
    explanation = get_explanation(text, label, confidence)
    print(f"预测标签: {label}")
    print(f"置信度: {confidence:.4f}")
    print(f"解释: {explanation}")