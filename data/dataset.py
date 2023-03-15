from typing import List, Any
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data: List[Any]):
        self.data = data

    def __getitem__(self, item):
        return str(self.data[item].text_a), int(self.data[item].label)

    def __len__(self):
        return len(self.data)

class TxtDataset(Dataset):
    def __init__(self, path: str):
        super(TxtDataset, self).__init__()
        with open(path) as fin:
            self.lines = fin.readlines()
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        temp = line.split('\t')
        sentence = temp[0]
        label = int(temp[1])
        return sentence, label
