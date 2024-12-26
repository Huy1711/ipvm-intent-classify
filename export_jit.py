import torch
from transformers import AutoTokenizer
from intent_classify.models import DistillBERTClassifier

CHECKPOINT_PATH = "lightning_logs/version_0/epoch7.ckpt"
SAVE_PATH = "lightning_logs/version_0/distilbert-7k.jit"

def save_model(model):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    example = "tell me about IPVM"
    example = tokenizer(example, return_tensors="pt")
    torch.jit.trace(model, (example["input_ids"], example["attention_mask"])).save(SAVE_PATH)

def load_torch_model():
    model = DistillBERTClassifier(num_classes=5)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))["state_dict"]["model"]
    )
    model.eval()
    return model

def test_output():
    """
    Ensure the original pytorch model and
    the exported jit model have the same output
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    jit_model = torch.jit.load(SAVE_PATH)
    torch_model = load_torch_model()

    example_data = tokenizer("tell me about IPVM", return_tensors="pt")

    jit_output = jit_model(**example_data)
    torch_output = torch_model(**example_data)

    assert torch.allclose(
        jit_output, torch_output
    ), "Pytorch model output and TorchScript model output are not the same"

if __name__ == "__main__":
    model = load_torch_model()
    save_model(model)
    test_output()
    
