import json
import torch
from transformers import AutoTokenizer
from intent_classify.models import DistillBERTClassifier
from intent_classify.utils import load_dataset, convert_intent_label_to_id, id_to_label_dict


CHECKPOINT_PATH = "checkpoints/epoch7.ckpt"
EVAL_SET_PATH = "data/139_sentences_eval.jsonl"
PREDICT_SAVE_PATH = "data/139_sentences_predict.jsonl"

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

    inference_data = load_dataset(EVAL_SET_PATH)

    predictions = []
    for data in inference_data:
        text = data["text"]
        inputs = tokenizer(text, return_tensors="pt")
        prediction = id_to_label_dict[predict(inputs)]
        predictions.append({
            "text": text,
            "pred": prediction,
        })

    with open(PREDICT_SAVE_PATH, "w") as f:
        for data in predictions:
            f.write(json.dumps(data) + "\n")
    print(f"Successfully saved inference result to {PREDICT_SAVE_PATH}")