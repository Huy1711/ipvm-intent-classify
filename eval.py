import tqdm
import time
import torch
import spacy
from transformers import AutoTokenizer
from intent_classify.models import DistillBERTClassifier
from intent_classify.utils import load_dataset, convert_intent_label_to_id, id_to_label_dict


CHECKPOINT_PATH = "lightning_logs/version_7/epoch17.ckpt"
EVAL_SET_PATH = [
    "data/139_sentences_eval.jsonl",
    "data/chatgpt-161-eval.jsonl",
    "data/claude-100-eval.jsonl",
]
# EVAL_SET_PATH = ["data/error_eval.jsonl"]


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
    # model = torch.jit.load(CHECKPOINT_PATH)
    model.eval()
    nlp = spacy.load('/Users/huynd/recommendation-v2-api/model')

    eval_data = load_dataset(EVAL_SET_PATH)
    eval_data = convert_intent_label_to_id(eval_data)

    wrong = 0
    predictions = []
    wrong_preds = []
    for data in tqdm.tqdm(eval_data):
        text = data["text"].replace("\n", " ")
        label = id_to_label_dict[data["label"]]
        start = time.time()
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        distilbert_prediction = id_to_label_dict[predict(inputs)]
        print(f"Prediction took {time.time() - start} seconds")

        doc = nlp(text)
        spacy_prediction = max(doc.cats, key=doc.cats.get)
        
        if spacy_prediction != label:
            wrong_preds.append([text, spacy_prediction, label])
            wrong += 1

        predictions.append([text.strip(), label, distilbert_prediction, spacy_prediction])
    predictions.sort(key=lambda x: x[1])
    with open("test_result.tsv", "w") as f:
        for pred in predictions:
            f.write("\t".join(pred) + "\n")

    for text, prediction, label in wrong_preds:
        print(f"Sentence: {text}")
        print("Predict:", prediction, "| Ground truth:", label)
        print("-"*80)
    print(f"Accuracy: {(len(eval_data) - wrong) * 100 / len(eval_data)}%")
    print(f"Wrong predictions: {wrong}")