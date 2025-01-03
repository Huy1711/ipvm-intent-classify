import torch
from torch.utils.data import Dataset

from .utils import convert_intent_label_to_id, load_dataset
from .augment import WordLevelAugment, CharacterLevelAugment

class TextClassificationDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len, padding, augment):
        self.dataset = load_dataset(filepath)
        self.dataset = convert_intent_label_to_id(self.dataset)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding
        self.augment = augment

        if self.augment:
            self.word_augment = WordLevelAugment(prob=0.5, aug_max=10, device="cuda:0")
            self.char_augment = CharacterLevelAugment(prob=0.5, aug_char_max=10)

    def __getitem__(self, index):
        data = self.dataset[index]
        text = data["text"]
        label = data["label"]

        if self.augment:
            text = self.word_augment.apply(text)
            text = self.char_augment.apply(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            max_length=self.max_len,
            padding=self.padding,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.dataset)
