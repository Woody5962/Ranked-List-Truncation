import pickle
import torch as t
import numpy as np
from torch.utils import data


BM25_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/BM25_results'
DRMM_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/drmm_results'
DRMM_TKS_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/drmm_tks_results'
STATS_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/statics'
GT_PATH = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/robust04_data/robust04_gt.pkl'


class Rank_Dataset(data.Dataset):
    def __init__(self, dataset_name: str, split: int):
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prepare(dataset_name, split)

    def data_prepare(self, dataset_name: str, split: int):
        if dataset_name == 'bm25':
            with open('{}/split_{}/BM25_train_s{}.pkl'.format(BM25_BASE, split, split), 'rb') as f:
                train_data_raw = pickle.load(f)
            with open('{}/split_{}/BM25_test_s{}.pkl'.format(BM25_BASE, split, split), 'rb') as f:
                test_data_raw = pickle.load(f)
            with open('{}/attncut_bm25_input.pkl'.format(STATS_BASE), 'rb') as f:
                stats_bm25 = pickle.load(f)

            X_train, X_test, y_train, y_test = [], [], [], []
            for key in train_data_raw:
                scores = np.array([train_data_raw[key]['retrieved_documents'][i]['norm_bm25_score'] 
                          for i in range(300)])
                stats = np.array(stats_bm25[key])
                input_features = np.column_stack((scores, stats))
                is_rel = [1 if train_data_raw[key]['retrieved_documents'][i]['is_relevant'] else 0 
                          for i in range(300)]
                X_train.append(input_features.tolist())
                y_train.append(is_rel)
            for key in test_data_raw:
                scores = np.array([test_data_raw[key]['retrieved_documents'][i]['norm_bm25_score'] 
                          for i in range(300)])
                stats = np.array(stats_bm25[key])
                input_features = np.column_stack((scores, stats))
                is_rel = [1 if test_data_raw[key]['retrieved_documents'][i]['is_relevant'] else 0 
                          for i in range(300)]
                X_test.append(input_features.tolist())
                y_test.append(is_rel)

        elif dataset_name == 'drmm':
            with open('{}/split_{}/drmm_train_s{}.pkl'.format(DRMM_BASE, split, split), 'rb') as f:
                train_data_raw = pickle.load(f)
            with open('{}/split_{}/drmm_test_s{}.pkl'.format(DRMM_BASE, split, split), 'rb') as f:
                test_data_raw = pickle.load(f)
            with open('{}/attncut_drmm_input.pkl'.format(STATS_BASE), 'rb') as f:
                stats_drmm = pickle.load(f)
            with open(GT_PATH, 'rb') as f:
                gt = pickle.load(f)
                for key in gt: gt[key] = set(gt[key])

            X_train, X_test, y_train, y_test = [], [], [], []
            for key in train_data_raw:
                scores = np.array([train_data_raw[key][i]['score'] for i in range(300)])
                stats = np.array(stats_drmm[key])
                input_features = np.column_stack((scores, stats))
                is_rel = [1 if train_data_raw[key][i]['doc_id'] in gt[key] else 0 
                          for i in range(300)]
                X_train.append(input_features.tolist())
                y_train.append(is_rel)
            for key in test_data_raw:
                scores = np.array([test_data_raw[key][i]['score'] for i in range(300)])
                stats = np.array(stats_drmm[key])
                input_features = np.column_stack((scores, stats))
                is_rel = [1 if test_data_raw[key][i]['doc_id'] in gt[key] else 0 
                          for i in range(300)]
                X_test.append(input_features.tolist())
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

        return t.Tensor(X_train), t.Tensor(X_test), t.Tensor(y_train), t.Tensor(y_test)

    def getX_train(self):
        return self.X_train

    def getX_test(self):
        return self.X_test

    def gety_train(self):
        return self.y_train

    def gety_test(self):
        return self.y_test


def dataloader(dataset_name: str, split: int, batch_size: int=20):
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
    a, b, c = dataloader('bm25', 1)
    
