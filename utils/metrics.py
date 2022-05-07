import numpy as np
import math
from numpy.lib.financial import irr
import torch as t
from sklearn import metrics

DCG_coef_300 = [math.log(j+2, 2) for j in range(300)]

class Metric:
    """通过前k个doc进行metric计算，k应该已经是数目了，而非坐标
    """
    def __init__(self):
        pass
    
    @classmethod
    def f1(cls, labels: np.array, k_s: list):
        N_D = np.sum(labels, axis=1)
        p_k, r_k, results = [], [], []
        for i in range(len(labels)): 
            count = np.sum(labels[i, :k_s[i]])
            p_k.append((count / k_s[i]))
            r_k.append((count / N_D[i]) if N_D[i] != 0 else 0)
            results.append((2 * p_k[-1] * r_k[-1] / (p_k[-1] + r_k[-1])) if p_k[-1] + r_k[-1] != 0 else 0)
        return np.mean(results)
    
    @classmethod
    def dcg(cls, labels: np.array, k_s: list, penalty=-1):
        results = []
        for i in range(len(labels)):
            label = labels[i, :k_s[i]]
            dcg_coef = DCG_coef_300[:k_s[i]]
            rele = (label == 1).astype(float)
            irre = (label != 1).astype(float)
            value = (rele / dcg_coef + penalty * irre / dcg_coef).sum()
            # for j in range(k_s[i]): 
            #     value += (1 / math.log(j+2, 2)) if x[j] else (penalty / math.log(j+2, 2))
            results.append(value)
        return np.mean(results)
    
    @classmethod
    def taskr_metric(cls, labels: np.array, predictions: np.array):
        """计算重排任务的metric，使用排序学习中常用的DCG（F1在全序列场景下无意义）

        Args:
            labels (np.array): labels
            predictions (np.array): 模型输出

        Returns:
            float: DCG
        """
        DCG_batch = []
        for sample_pred, sample_label in zip(predictions, labels):
            DCG_sample = 0
            sort_index = np.argsort(-sample_pred)
            for i, origin_index in enumerate(sort_index):
                DCG_sample += ((1 / math.log2(i+2)) if sample_label[origin_index] else (-1 / math.log2(i+2)))
            DCG_batch.append(DCG_sample)
        return np.mean(DCG_batch)
    
    @classmethod
    def taskc_metric(cls, labels: np.array, predictions: np.array):
        """计算分类任务的metric，使用AUC

        Args:
            labels (np.array): labels
            predictions (np.array): 模型输出

        Returns:
            float: 整个batch docs的AUC
        """
        tmp_auc, count_auc = 0, 0
        for i in range(labels.shape[0]):
            if sum(labels[i]) == 0 or sum(labels[i]) == len(labels[i]): continue
            tmp_auc += metrics.roc_auc_score(y_true=labels[i], y_score=predictions[i])
            count_auc += 1
        return tmp_auc / count_auc


class Metric_for_Loss:
    """通过前k个doc进行metric计算，k应该已经是数目了，而非坐标
    """
    def __init__(self) -> None:
        pass
    
    @classmethod
    def f1(cls, label: t.Tensor, k: int):
        N_D = t.sum(label)
        count = t.sum(label[:k])
        p_k = count.div(k)
        r_k = count.div(N_D) if N_D != 0 else t.tensor(0)
        return p_k.mul(r_k).mul(2).div(p_k.add(r_k)) if p_k.add(r_k) != 0 else t.tensor(0)
    
    @classmethod
    def dcg(cls, label: t.Tensor, k: int, penalty: int=-1):
        rele = (label[:k] == 1.).float()
        irre = (label[:k] != 1.).float()
        dcg_coef = t.tensor(DCG_coef_300[:k])
        value = rele.div(dcg_coef).add(irre.div(dcg_coef).mul(penalty)).sum()
        # for i in range(k):
        #     value = value.add(1 / math.log(i+2, 2)) if label[i] == 1 else value.add(penalty / math.log(i+2, 2))
        return value


if __name__ == '__main__':
    x = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]])
    k_s = np.array([1, 2, 1])
    r1 = Metric.f1(x, k_s)
    r2 = Metric.dcg(x, k_s)
    print(r1, r2)