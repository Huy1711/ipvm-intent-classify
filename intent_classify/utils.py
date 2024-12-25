import json

intent_label_to_id_dict = {
    "general_information": 0,
    "account_help": 1,
    "troubleshoot_product": 2,
    "lookup_report": 3,
    "lookup_person": 4,
}

id_to_label_dict = {
    0: "general_information",
    1: "account_help",
    2: "troubleshoot_product",
    3: "lookup_report",
    4: "lookup_person",
}


def load_dataset(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_intent_label_to_id(data):
    dataset = []
    for sent in data:
        dataset.append(
            {
                "text": sent["text"],
                "label": intent_label_to_id_dict[sent["label"]],
            }
        )
    return dataset


def compute_accuracy(idx, targets):
    n_correct = (idx == targets).sum().item()
    accuracy = (n_correct * 100.0) / targets.size(0)
    return accuracy
