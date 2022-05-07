import pickle
import torch as t
import numpy as np
from torch.utils import data

# import sys
# sys.path.append('../')
# from utils.batchnorm import batch_norm

DATASET_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/dataset'


class Rank_Dataset(data.Dataset):
    def __init__(self, retrieve_data: str='robust04', dataset_name: str='bm25'):
        self.database = DATASET_BASE + '/' + retrieve_data
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prepare(dataset_name)
        # self.X, train_size = t.cat((self.X_train, self.X_test), dim=0), self.X_train.shape[0]
        # self.X_norm = batch_norm(self.X)
        # self.X_train_norm, self.X_test_norm = self.X_norm[:train_size], self.X_norm[train_size:]

    def data_prepare(self, dataset_name: str):
        """根据dataset name制作数据集

        Args:
            dataset_name (str): bm25, drmm, drmm_tks

        Returns:
            [type]: [description]
        """
        with open('{}/{}_train.pkl'.format(self.database, dataset_name), 'rb') as f:
            train_data_raw = pickle.load(f)
        with open('{}/{}_test.pkl'.format(self.database, dataset_name), 'rb') as f:
            test_data_raw = pickle.load(f)
        with open('{}/attncut/{}_train.pkl'.format(self.database, dataset_name), 'rb') as f:
            stats_train = pickle.load(f)
        with open('{}/attncut/{}_test.pkl'.format(self.database, dataset_name), 'rb') as f:
            stats_test = pickle.load(f)
        with open('{}/gt.pkl'.format(self.database), 'rb') as f:
            gt = pickle.load(f)
            for key in gt: gt[key] = set(gt[key])

        X_train, X_test, y_train, y_test = [], [], [], []
        for key in train_data_raw:
            scores = np.array(list(train_data_raw[key].values()))
            stats = np.array(stats_train[key])
            input_features = np.column_stack((scores, stats))
            is_rel = list(map(lambda x: 1 if x in gt[key] else 0, train_data_raw[key].keys()))
            X_train.append(input_features.tolist())
            y_train.append(is_rel)

        for key in test_data_raw:
            scores = np.array(list(test_data_raw[key].values()))
            stats = np.array(stats_test[key])
            input_features = np.column_stack((scores, stats))
            is_rel = list(map(lambda x: 1 if x in gt[key] else 0, test_data_raw[key].keys()))
            X_test.append(input_features.tolist())
            y_test.append(is_rel)

        return t.Tensor(X_train), t.Tensor(X_test), t.Tensor(y_train), t.Tensor(y_test)

    def getX_train(self):
        return self.X_train

    def getX_test(self):
        return self.X_test

    def gety_train(self):
        return self.y_train

    def gety_test(self):
        return self.y_test


def dataloader(retrieve_data: str='robust04', dataset_name: str='bm25', batch_size: int=20):
	"""
	batch_ratio: batchsize / datasize
	"""
	rank_data = Rank_Dataset(retrieve_data, dataset_name)

	X_train = rank_data.getX_train()
	X_test = rank_data.getX_test()
	y_train = rank_data.gety_train()
	y_test = rank_data.gety_test()

	train_dataset = data.TensorDataset(X_train, y_train)
	test_dataset = data.TensorDataset(X_test, y_test)
	train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	return train_loader, test_loader, rank_data


if __name__ == '__main__':
    a, b, c = dataloader(retrieve_data='mq2007')
    xtr = c.getX_train()
    xte = c.getX_test()
    ytr = c.gety_train()
    yte = c.gety_test()
    pass
