import pickle
import os
import torch as t
import numpy as np
from torch.utils import data


STATS_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/statics'


class Rank_Dataset(data.Dataset):
    def __init__(self, dataset_name: str, split: int, stage: str='train'):
        self.base_dir = '{}/bicut_{}_s{}_{}/'.format(STATS_BASE, dataset_name, split, stage)
        self.index_list = os.listdir(self.base_dir)

    def __getitem__(self, index):
        _sample_dir = '{}/{}'.format(self.base_dir ,self.index_list[index])
        with open(_sample_dir, 'rb') as f: _sample = pickle.load(f)
        return t.from_numpy(_sample[0]).float(), t.Tensor(_sample[1])

    def __len__(self):
        return len(self.index_list)

def dataloader(dataset_name: str, split: int, batch_size: int=20):
    """dataloader for bicut

    Args:
        dataset_name (str): [description]
        split (int): [description]
        batch_size (int, optional): batch_size. Defaults to 20.

    Returns:
        [type]: [description]
    """
    train_dataset = Rank_Dataset(dataset_name=dataset_name, split=split, stage='train') 
    test_dataset = Rank_Dataset(dataset_name=dataset_name, split=split, stage='test')

    train_loader = data.DataLoader(dataset=train_dataset,
            batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size,
            shuffle=True, pin_memory=True, num_workers=4)

    return train_loader, test_loader 


if __name__ == '__main__':
    a, b, c = dataloader('BM25', 1)
