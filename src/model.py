from transformers import AutoModelForSequenceClassification

def get_model(model_name="vinai/bertweet-base"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    return model