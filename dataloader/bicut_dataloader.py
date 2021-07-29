import pickle
import os
import torch as t
from torch.utils import data


DATASET_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/dataset'


class Rank_Dataset(data.Dataset):
    def __init__(self, dataset_name: str, stage: str='train'):
        self.base_dir = '{}/bicut/{}_{}/'.format(DATASET_BASE, dataset_name, stage)
        self.index_list = os.listdir(self.base_dir)

        with open('{}/robust04_gt.pkl'.format(DATASET_BASE), 'rb') as f:
            self.gt = pickle.load(f)
            for key in self.gt: self.gt[key] = set(self.gt[key])
        with open('{}/{}_{}.pkl'.format(DATASET_BASE, dataset_name, stage), 'rb') as f:
            self.data_raw = pickle.load(f)

    def __getitem__(self, index):
        _sample_dir = '{}/{}'.format(self.base_dir, self.index_list[index])
        qid = self.index_list[index].split('.')[0]
        with open(_sample_dir, 'rb') as f: _sample = pickle.load(f)
        _label = list(map(lambda x: 1 if x in self.gt[qid] else 0, self.data_raw[qid].keys()))
        return t.tensor(_sample).float(), t.Tensor(_label)

    def __len__(self):
        return len(self.index_list)

def dataloader(dataset_name: str, batch_size: int=20, num_workers: int=8):
    """dataloader for bicut

    Args:
        dataset_name (str): [description]
        split (int): [description]
        batch_size (int, optional): batch_size. Defaults to 20.

    Returns:
        [type]: [description]
    """
    train_dataset = Rank_Dataset(dataset_name=dataset_name, stage='train') 
    test_dataset = Rank_Dataset(dataset_name=dataset_name, stage='test')

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader 


if __name__ == '__main__':
    a, b, c = dataloader('BM25', 1)
    pass