import torch as t
import torch.nn as nn

def batch_norm(data: t.Tensor) -> t.Tensor:
    feature_nums = data.shape[2]
    data_norm = t.zeros_like(data)
    bn = nn.BatchNorm1d(1, affine=False, track_running_stats=False)
    for i in range(feature_nums):
        feature_batch = data[:,:,i]
        data_norm[:,:,i] = bn(feature_batch)
    return data_norm

if __name__ == '__main__':
    a = t.tensor([[[1,2.]],[[2,2.5]],[[3,4.]]])
    print(batch_norm(a))
    pass
