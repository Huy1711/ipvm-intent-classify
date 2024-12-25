import torch.nn as nn
from transformers import DistilBertModel


class DistillBERTClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=768, dropout=0.3):
        super(DistillBERTClassifier, self).__init__()
        self.encoder = DistilBertModel.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        self.head = ClassificationHead(
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output[0]
        output = hidden_state[:, 0]
        output = self.head(output)
        return output


class ClassificationHead(nn.Module):
    def __init__(self, output_dim, hidden_dim=768, dropout=0.3):
        super(ClassificationHead, self).__init__()
        self.pre_classifier = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        pooler = self.pre_classifier(x)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
