import json
import tqdm
import torch
from transformers import AutoTokenizer
from intent_classify.models import DistillBERTClassifier
from intent_classify.utils import load_dataset, convert_intent_label_to_id, id_to_label_dict


CHECKPOINT_PATH = "lightning_logs/version_0/epoch7.ckpt"
EVAL_SET_PATH = "data/7k_sentences_train.jsonl"


def predict(inputs):
    with torch.no_grad():
        logits = model(**inputs.to("cpu"))
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    model = DistillBERTClassifier(num_classes=5)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))["state_dict"]["model"]
    )
    model.eval()

    eval_data = load_dataset(EVAL_SET_PATH)
    eval_data = convert_intent_label_to_id(eval_data)

    wrong = 0
    predictions = []
    wrong_preds = []
    for data in tqdm.tqdm(eval_data):
        text = data["text"]
        label = id_to_label_dict[data["label"]]
        inputs = tokenizer(text, return_tensors="pt")
        prediction = id_to_label_dict[predict(inputs)]
        if prediction != label:
            wrong_preds.append([text, prediction, label])
            wrong += 1

    for text, prediction, label in wrong_preds:
        print(f"Sentence: {text}")
        print("Predict:", prediction, "| Ground truth:", label)
        print("-"*80)
    print(f"Accuracy: {(len(eval_data) - wrong) * 100 / len(eval_data)}%")