import pickle
import torch as t
from torch.utils import data

import sys
sys.path.append('../')
from utils.batchnorm import batch_norm


BM25_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/BM25_results'
DRMM_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/drmm_results'
DRMM_TKS_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/drmm_tks_results'
GT_PATH = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/robust04_data/robust04_gt.pkl'


class Rank_Dataset(data.Dataset):
    def __init__(self, dataset_name: str, split: int):
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prepare(dataset_name, split)
        self.X, train_size = t.cat((self.X_train, self.X_test), dim=0), self.X_train.shape[0]
        self.X_norm = batch_norm(self.X)
        self.X_train_norm, self.X_test_norm = self.X_norm[:train_size], self.X_norm[train_size:]

    def data_prepare(self, dataset_name: str, split: int):
        if dataset_name == 'bm25':
            with open('{}/split_{}/BM25_train_s{}.pkl'.format(BM25_BASE, split, split), 'rb') as f:
                train_data_raw = pickle.load(f)
            with open('{}/split_{}/BM25_test_s{}.pkl'.format(BM25_BASE, split, split), 'rb') as f:
                test_data_raw = pickle.load(f)

            X_train, X_test, y_train, y_test = [], [], [], []
            for key in train_data_raw:
                scores = [train_data_raw[key]['retrieved_documents'][i]['norm_bm25_score'] 
                          for i in range(300)]
                is_rel = [1 if train_data_raw[key]['retrieved_documents'][i]['is_relevant'] else 0 
                          for i in range(300)]
                X_train.append(scores)
                y_train.append(is_rel)
            for key in test_data_raw:
                scores = [test_data_raw[key]['retrieved_documents'][i]['norm_bm25_score'] 
                          for i in range(300)]
                is_rel = [1 if test_data_raw[key]['retrieved_documents'][i]['is_relevant'] else 0 
                          for i in range(300)]
                X_test.append(scores)
                y_test.append(is_rel)

        elif dataset_name == 'drmm':
            with open('{}/split_{}/drmm_train_s{}.pkl'.format(DRMM_BASE, split, split), 'rb') as f:
                train_data_raw = pickle.load(f)
            with open('{}/split_{}/drmm_test_s{}.pkl'.format(DRMM_BASE, split, split), 'rb') as f:
                test_data_raw = pickle.load(f)
            with open(GT_PATH, 'rb') as f:
                gt = pickle.load(f)
                for key in gt: gt[key] = set(gt[key])

            X_train, X_test, y_train, y_test = [], [], [], []
            for key in train_data_raw:
                scores = [train_data_raw[key][i]['score'] for i in range(300)]
                is_rel = [1 if train_data_raw[key][i]['doc_id'] in gt[key] else 0 
                          for i in range(300)]
                X_train.append(scores)
                y_train.append(is_rel)
            for key in test_data_raw:
                scores = [test_data_raw[key][i]['score'] 
                          for i in range(300)]
                is_rel = [1 if test_data_raw[key][i]['doc_id'] in gt[key] else 0 
                          for i in range(300)]
                X_test.append(scores)
                y_test.append(is_rel)

        elif dataset_name == 'drmm_tks':
            with open('{}/split_{}/drmm_tks_train_s{}.pkl'.format(DRMM_TKS_BASE, split, split), 'rb') as f:
                train_data_raw = pickle.load(f)
            with open('{}/split_{}/drmm_tks_test_s{}.pkl'.format(DRMM_TKS_BASE, split, split), 'rb') as f:
                test_data_raw = pickle.load(f)
            with open(GT_PATH, 'rb') as f:
                gt = pickle.load(f)
                for key in gt: gt[key] = set(gt[key])

            X_train, X_test, y_train, y_test = [], [], [], []
            for key in train_data_raw:
                scores = [train_data_raw[key][i]['score'] 
                          for i in range(300)]
                is_rel = [1 if train_data_raw[key][i]['doc_id'] in gt[key] else 0 
                          for i in range(300)]
                X_train.append(scores)
                y_train.append(is_rel)
            for key in test_data_raw:
                scores = [test_data_raw[key][i]['score'] 
                          for i in range(300)]
                is_rel = [1 if test_data_raw[key][i]['doc_id'] in gt[key] else 0 
                          for i in range(300)]
                X_test.append(scores)
                y_test.append(is_rel)

        X_train, X_test = t.unsqueeze(t.Tensor(X_train), dim=1).permute(0, 2, 1), t.unsqueeze(t.Tensor(X_test), dim=1).permute(0, 2, 1)
        y_train, y_test = t.Tensor(y_train), t.Tensor(y_test)
        
        d00, d01, _ = X_train.shape
        d10, d11, _ = X_test.shape
        position_embedding_train, position_embedding_test = t.randn(d00, d01, 127), t.randn(d10, d11, 127)
        X_train = t.cat((X_train, position_embedding_train), dim=2)
        X_test = t.cat((X_test, position_embedding_test), dim=2)

        return X_train, X_test, y_train, y_test

    def getX_train(self):
        return self.X_train

    def getX_test(self):
        return self.X_test

    def gety_train(self):
        return self.y_train

    def gety_test(self):
        return self.y_test


def dataloader(dataset_name: str, split: int, batch_size: int):
	"""
	batch_ratio: batchsize / datasize
	"""
	rank_data = Rank_Dataset(dataset_name=dataset_name, split=split)

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
    a, b, c = dataloader('drmm_tks', 1, 32)
    xtr = c.getX_train()
    xte = c.getX_test()
    ytr = c.getX_train()
    yte = c.gety_test()
    pass