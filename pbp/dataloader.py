import torch
from torch.utils.data import DataLoader, Sampler
from operator import itemgetter

class LengthBasedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.lengths = [len(item['sequence']) for item in dataset]
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda k: self.lengths[k])

    def __iter__(self):
        batches = []
        current_batch = []
        for idx in self.sorted_indices:
            current_batch.append(idx)
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    # each example has a "home_embedding" and a "away_embedding" we need to pass as a tuple to the model
    home_embeddings = torch.stack([torch.tensor(item['home_embedding'], dtype=torch.bfloat16) for item in batch])
    away_embeddings = torch.stack([torch.tensor(item['away_embedding'], dtype=torch.bfloat16) for item in batch])
    
    # Labels are identical to input_ids
    labels = input_ids.clone()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'teams': (home_embeddings, away_embeddings)
    }

def tokenize_collate(tokenizer):
    def collate_fn(batch):
        sequences = [item['sequence'] for item in batch]
        tokenized = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
   
    return collate_fn 


