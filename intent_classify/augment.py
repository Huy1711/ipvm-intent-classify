import random
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

class CharacterLevelAugment(object):
    def __init__(self, prob=0.3, aug_char_max=1, random_action="swap"):
        keyboard_aug = nac.KeyboardAug(aug_char_max=aug_char_max)
        random_aug = nac.RandomCharAug(action=random_action, aug_char_max=aug_char_max)
        self.augment_list = [keyboard_aug, random_aug]
        self.prob = prob

    def apply(self, text):
        if random.random() > self.prob:
            return text
        aug = random.choice(self.augment_list)
        return aug.augment(text)


class WordLevelAugment(object):
    def __init__(self, prob=0.3, aug_max=3, device="cpu"):
        context_aug = naw.ContextualWordEmbsAug(
            model_path='roberta-base', 
            action="substitute", 
            aug_max=aug_max,
            device=device
        )
        back_translation_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de', 
            to_model_name='facebook/wmt19-de-en',
            device=device
        )
        self.augment_list = [context_aug, back_translation_aug]
        self.prob = prob

    def apply(self, text):
        if random.random() > self.prob:
            return text
        aug = random.choice(self.augment_list)
        return aug.augment(text)