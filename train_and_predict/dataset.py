from torch.utils.data import Dataset, DataLoader
import torch

class PowerDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = [torch.tensor(inp, dtype=torch.float32) for inp in inputs]
        self.targets = [torch.tensor(tgt, dtype=torch.float32) for tgt in targets]
        self.lengths = [len(inp) for inp in inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.lengths[idx]
    
def collate_fn(batch):
    inputs, targets, lengths = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)
    return inputs, targets, lengths