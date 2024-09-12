import torch

def tokenize_collate(tokenizer):
    def collate_fn(batch):
        max_len = max([len(item["text"]) for item in batch])
        max_len = min(max_len, 8192)
        inputs = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        labels = inputs.input_ids.clone()
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels
        }
    return collate_fn
