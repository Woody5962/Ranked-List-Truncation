import numpy as np
import math
import torch as t


class Metric:
    def __init__(self):
        pass
    
    @classmethod
    def f1(cls, labels: np.array, k_s: list):
        N_D = np.sum(labels, axis=1)
        p_k, r_k, results = [], [], []
        for i in range(len(labels)): 
            count = np.sum(labels[i, :k_s[i]])
            p_k.append((count / k_s[i]) if k_s[i] != 0 else 0)
            r_k.append((count / N_D[i]) if N_D[i] != 0 else 0)
            results.append((2 * p_k[-1] * r_k[-1] / (p_k[-1] + r_k[-1])) if p_k[-1] + r_k[-1] != 0 else 0)
        return sum(results) / len(results)
    
    @classmethod
    def dcg(cls, labels: np.array, k_s: list, penalty=-1):
        def dcg_line(x, k):
            value = 0
            for i in range(k): 
                value += (1 / math.log(i+2, 2)) if x[i] else (penalty / math.log(i+2, 2))
            return value
        results = []
        for i in range(len(labels)):
            results.append(dcg_line(labels[i], k_s[i]))
        return sum(results) / len(results)


if __name__ == '__main__':
    x = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]])
    k_s = np.array([1, 2, 1])
    r1 = Metric.f1(x, k_s)
    r2 = Metric.dcg(x, k_s)
    print(r1, r2)


class Metric_for_Loss:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def f1(cls, label: t.Tensor, k: int):
        N_D = t.sum(label)
        count = t.sum(label[:k])
        p_k = t.div(count, k) if k != 0 else t.tensor(0)
        r_k = t.div(count, N_D) if N_D != 0 else t.tensor(0)
        return t.div(t.mul(p_k, r_k).mul(2), t.add(p_k, r_k)) if t.add(p_k, r_k) != 0 else t.tensor(0)
    
    @classmethod
    def dcg(cls, label: t.Tensor, k: int, penalty: int=-1):
        value = t.tensor(0)
        for i in range(k):
            value = t.add(value, (1 / math.log(i+2, 2)) if label[i] == 1 else (penalty / math.log(i+2, 2)))
        return value