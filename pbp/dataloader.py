import torch

def tokenize_collate(tokenizer):
    def collate_fn(batch):
        inputs = [item["input_ids"] for item in batch]
        max_len = max(input.shape[1] for input in inputs)
        # pad all inputs to the same length using the tokenizer's pad_token_id
        padded_inputs = torch.stack([
            torch.cat([input, torch.tensor([tokenizer.pad_token_id] * (max_len - input.shape[1]))])
            for input in inputs
        ])
        attention_mask = (padded_inputs != tokenizer.pad_token_id).long()
        
        labels = padded_inputs.clone()
        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate_fn
