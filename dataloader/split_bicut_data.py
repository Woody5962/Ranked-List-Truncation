import pickle
import numpy as np
import os


BM25_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/BM25_results'
DRMM_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/drmm_results'
DRMM_TKS_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/my_results/drmm_tks_results'
STATS_BASE = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/statics'
GT_PATH = '/home/LAB/wangd/graduation_project/ranked list truncation/data_prep/robust04_data/robust04_gt.pkl'


def data_prepare(dataset_name: str, split: int):
    if dataset_name == 'BM25':
        with open('{}/split_{}/BM25_train_s{}.pkl'.format(BM25_BASE, split, split), 'rb') as f:
            train_data_raw = pickle.load(f)
        with open('{}/split_{}/BM25_test_s{}.pkl'.format(BM25_BASE, split, split), 'rb') as f:
            test_data_raw = pickle.load(f)

        for key in train_data_raw:
            scores = np.array([train_data_raw[key]['retrieved_documents'][i]['norm_bm25_score']
                        for i in range(300)])
            stats = np.array(stats_bm25[key])
            input_features = np.column_stack((scores, stats))
            is_rel = [1 if train_data_raw[key]['retrieved_documents'][i]['is_relevant'] else 0
                        for i in range(300)]
            if not os.path.exists('{}/bicut_bm25_s{}_train/'.format(STATS_BASE, split)): 
                os.mkdir('{}/bicut_bm25_s{}_train/'.format(STATS_BASE, split))
            with open('{}/bicut_bm25_s{}_train/qid_{}.pkl'.format(STATS_BASE, split, key), 'wb') as f:
                pickle.dump((input_features, is_rel), f)
            print('s_{}_train_{}'.format(split, key))

        for key in test_data_raw:
            scores = np.array([test_data_raw[key]['retrieved_documents'][i]['norm_bm25_score']
                        for i in range(300)])
            stats = np.array(stats_bm25[key])
            input_features = np.column_stack((scores, stats))
            is_rel = [1 if test_data_raw[key]['retrieved_documents'][i]['is_relevant'] else 0
                        for i in range(300)]
            if not os.path.exists('{}/bicut_bm25_s{}_test/'.format(STATS_BASE, split)): 
                os.mkdir('{}/bicut_bm25_s{}_test/'.format(STATS_BASE, split))
            with open('{}/bicut_bm25_s{}_test/qid_{}.pkl'.format(STATS_BASE, split, key), 'wb') as f:
                pickle.dump((input_features, is_rel), f)
            print('s_{}_test_{}'.format(split, key))

    elif dataset_name == 'DRMM':
        with open('{}/split_{}/drmm_train_s{}.pkl'.format(DRMM_BASE, split, split), 'rb') as f:
            train_data_raw = pickle.load(f)
        with open('{}/split_{}/drmm_test_s{}.pkl'.format(DRMM_BASE, split, split), 'rb') as f:
            test_data_raw = pickle.load(f)
        with open(GT_PATH, 'rb') as f:
            gt = pickle.load(f)
            for key in gt:
                gt[key] = set(gt[key])

        for key in train_data_raw:
            scores = np.array([train_data_raw[key][i]['score'] for i in range(300)])
            stats = np.array(stats_drmm[key])
            input_features = np.column_stack((scores, stats))
            is_rel = [1 if train_data_raw[key][i]['doc_id'] in gt[key] else 0
                        for i in range(300)]
            if not os.path.exists('{}/bicut_drmm_s{}_train/'.format(STATS_BASE, split)): 
                os.mkdir('{}/bicut_drmm_s{}_train/'.format(STATS_BASE, split))
            with open('{}/bicut_drmm_s{}_train/qid_{}.pkl'.format(STATS_BASE, split, key), 'wb') as f:
                pickle.dump((input_features, is_rel), f)
            print('s_{}_train_{}'.format(split, key))
        
        for key in test_data_raw:
            scores = np.array([test_data_raw[key][i]['score'] for i in range(300)])
            stats = np.array(stats_drmm[key])
            input_features = np.column_stack((scores, stats))
            is_rel = [1 if test_data_raw[key][i]['doc_id'] in gt[key] else 0
                        for i in range(300)]
            if not os.path.exists('{}/bicut_drmm_s{}_test/'.format(STATS_BASE, split)): 
                os.mkdir('{}/bicut_drmm_s{}_test/'.format(STATS_BASE, split))
            with open('{}/bicut_drmm_s{}_test/qid_{}.pkl'.format(STATS_BASE, split, key), 'wb') as f:
                pickle.dump((input_features, is_rel), f)
            print('s_{}_test_{}'.format(split, key))


if __name__ == '__main__':
    # with open('{}/bicut_bm25_input.pkl'.format(STATS_BASE), 'rb') as f:
    #     stats_bm25 = pickle.load(f)
    with open('{}/bicut_drmm_input.pkl'.format(STATS_BASE), 'rb') as f:
        stats_drmm = pickle.load(f)
    print('stats is loaded successfully!')
    for i in range(1, 6):
        data_prepare('DRMM', i)
    print('bicut-drmm done!')