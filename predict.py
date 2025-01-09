import tqdm
import json
import torch
import spacy
from transformers import AutoTokenizer
from intent_classify.models import DistillBERTClassifier
from intent_classify.utils import load_dataset, id_to_label_dict


CHECKPOINT_PATH = "lightning_logs/version_0/epoch7.ckpt"
INFER_SET_PATH = [
    "data/error_eval.jsonl"
]
OUTPUT_PATH = "data/report_mistakes_distilbert.tsv"


def predict(inputs):
    with torch.no_grad():
        logits = model(**inputs.to("cpu"))
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    model = DistillBERTClassifier(num_classes=5)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))["state_dict"]["model"]
    )
    model.eval()

    nlp = spacy.load('/Users/huynd/recommendation-v2-api/model')

    data = load_dataset(INFER_SET_PATH)

    predictions = []
    for data in tqdm.tqdm(data):
        text = data["text"]
        
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        prob = predict(inputs)
        confidence_score = prob[0][prob.argmax().item()].item()
        prediction = id_to_label_dict[prob.argmax().item()]

        # doc = nlp(text)
        # prediction = max(doc.cats, key=doc.cats.get)
        # confidence_score = doc.cats[prediction]

        predictions.append({
            "text": text,
            "pred": prediction,
            "score": confidence_score,
        })

    # with open(OUTPUT_PATH, "w") as f:
    #     for pred in predictions:
    #         f.write(json.dumps(pred) + "\n")
            
    with open(OUTPUT_PATH, "w") as f:
        for pred in predictions:
            f.write("\t".join([pred["text"], pred["pred"]]) + "\n")
    