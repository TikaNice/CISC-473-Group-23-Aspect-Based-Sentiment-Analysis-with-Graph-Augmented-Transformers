import numpy as np
import spacy


class DependencyGraphBuilder:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def build(self, text, tokenizer, max_length=128):
        doc = self.nlp(text)
        words = [token.text for token in doc]

        # BERT word alignment: each word may become multiple subwords.
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        word_ids = encoded.word_ids()
        kept_ids = sorted({
            wid for wid in word_ids if wid is not None
        })

        if not kept_ids:
            raise ValueError("No words survived tokenization")

        # Map original spaCy word indices to GCN node indices.
        old_to_new = {
            old: new for new, old in enumerate(kept_ids)
        }

        n = len(kept_ids)
        adjacency = np.eye(n, dtype=np.float32)

        for token in doc:
            i = old_to_new.get(token.i)
            j = old_to_new.get(token.head.i)
            if i is not None and j is not None:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

        # Symmetric degree normalization: D^-1/2 A D^-1/2
        degree = adjacency.sum(axis=1)
        inv_sqrt = np.power(
            np.maximum(degree, 1.0), -0.5
        )
        adjacency = (
            inv_sqrt[:, None]
            * adjacency
            * inv_sqrt[None, :]
        )

        aligned_ids = [
            old_to_new.get(wid, -1)
            if wid is not None else -1
            for wid in word_ids
        ]

        return {
            "words": [words[i] for i in kept_ids],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "word_ids": aligned_ids,
            "adjacency": adjacency,
            "num_words": n,
        }
