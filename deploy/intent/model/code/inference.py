import os
import json
import torch
from transformers import AutoTokenizer

MODEL_FILE_NAME = 'model.pt'
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir, context):
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.to(device)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        inputs = tokenizer(data["text"], return_tensors="pt")
        return inputs
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        predictions = model(**input_data.to(device))
        predictions = torch.nn.functional.softmax(predictions, dim=1)
    return predictions


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return prediction.cpu().numpy().tolist()
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
