import torch as t
from torch import nn, unsqueeze
import torch.nn.functional as F
import math
import random

from .metrics import Metric_for_Loss


class BiCutLoss(nn.Module):
    """bicut对应的loss
    
    """
    def __init__(self, alpha: float=0.65, r: float=0.0971134020, metric: str='nci'):
        super().__init__()
        self.alpha = alpha
        self.r = r

    def slice_index(self, out_tensor):
        """
        0对应truncation，1对应continue
        """
        temp = t.argmax(out_tensor, dim=1)
        ones_tensor = t.ones_like(temp)
        if temp.equal(ones_tensor): return temp.shape[0]
        length = temp.shape[0] - 1
        return length - t.argmin(t.flip(temp, dims=[0]))
    
    def forward(self, output, labels):
        mask = t.ones_like(output)
        for i in range(mask.shape[0]):
            mask[i][self.slice_index(output[i]):] = 0
        r = t.ones_like(output)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if self.metric == 'nci': 
                    r[i][j] = t.tensor([0, -3.6 / math.log2(j+2)]) if labels[i][j] == 1 else t.tensor([0, self.alpha * 0.1])
                else: 
                    r[i][j] = t.tensor([(1 - self.alpha) / self.r, 0]) if labels[i][j] == 1 else t.tensor([0, self.alpha / (1 - self.r)])
        
        mask_output = output.mul(mask)
        loss_matrix = mask_output.mul(r)
        return t.div(t.sum(loss_matrix), output.shape[0])


class ChoopyLoss(nn.Module):
    """Choopy对应的loss

    """
    def __init__(self, metric: str='f1'):
        super().__init__()
        self.metric = metric
    
    def forward(self, output, labels):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j)
        
        output = output.squeeze()
        loss_matrix = output.mul(r)
        return -t.div(t.sum(loss_matrix), output.shape[0])


class AttnCutLoss(nn.Module):
    """AttnCut对应的loss

    """
    def __init__(self, metric: str='f1', tau: float=0.95):
        super().__init__()
        self.metric = metric
        self.tau = tau
    
    def forward(self, output: t.Tensor, labels: t.Tensor):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j)
        q = t.exp(t.div(r, self.tau))
        norm_factor = t.sum(q, axis=1).unsqueeze(dim=1)
        q = t.div(1, norm_factor)
        
        output_1 = t.log(output.squeeze())
        loss_matrix = output_1.mul(q)
        return -t.div(t.sum(loss_matrix), output.shape[0])
    
    
class RerankLoss(nn.Module):
    """
    Creates a criterion that measures rank hinge loss.
    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).
    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.
    The loss function for each sample in the mini-batch is:
    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """
    def __init__(self, margin: float = 1., reduction: str = 'mean'):
        """
        :class:`RankHingeLoss` constructor.
        :param margin: Margin between positive and negative scores.
            Float. Has a default value of :math:`0`.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, output: t.Tensor, labels: t.Tensor):
        """
        Calculate rank hinge loss.
        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Hinge loss computed by user-defined margin.
        """
        loss = []
        for sample_pred, sample_label in zip(output, labels):
            if t.sum(sample_label) == 0: return t.tensor(0)
            y_pos, y_neg = [], []
            n_pos, n_neg = 0, 0
            for i, label in enumerate(sample_label):
                if label: 
                    y_pos.append(sample_pred[i])
                    n_pos += 1
                else: 
                    y_neg.append(sample_pred[i])
                    n_neg += 1
            y_pos_1D = t.tensor(y_pos).unsqueeze(-1).expand(-1, n_neg).flatten()
            y_neg_1D = t.tensor(y_neg).repeat(n_pos)
            y_true = t.ones_like(y_pos_1D)
            loss.append(F.margin_ranking_loss(
                y_pos_1D, y_neg_1D, y_true,
                margin=self.margin,
                reduction=self.reduction
            ))
        return t.div(t.sum(t.tensor(loss)), output.shape[0])

    
class MTCutLoss(nn.Module):
    """对应于多任务学习的loss

    Args:
        nn ([type]): [description]
    """
    def __init__(self, metric: str='f1', tau: float=0.95):
        super().__init__()
        self.cutloss = AttnCutLoss(metric, tau)
        self.rerankloss = RerankLoss()
        
    def forward(self, output, labels):
        rerank_y, cut_y = output
        rerank_label, cut_label = labels, labels
        cutloss, rerankloss = self.cutloss(cut_y, cut_label), self.rerankloss(rerank_y, rerank_label)
        return cutloss + rerankloss
        
        
if __name__ == '__main__':
    a = t.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], requires_grad=True).unsqueeze(dim=2)
    # a_1 = t.tensor([[[0.1, 0.9], [0.2, 0.8], [0.4, 0.6]], [[0.12, 0.88], [0.8, 0.2], [0.4, 0.6]]])
    b = t.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    # bi_loss = BiCutLoss()
    # ch_loss = ChoopyLoss()
    at_loss = AttnCutLoss()
    # l1 = bi_loss(a_1, b)
    # l2 = ch_loss(a, b)
    print(a.grad)
    l3 = at_loss(a, b)
    l3.backward()
    print(a.grad)
    # print(l1, l2, l3)
    loss_f = RerankLoss()
    # y_pred = t.tensor([1., 1.2, 3.1, 0.1, 1.1, 4.2, 2.1, 5.1]).unsqueeze(dim=-1)
    # y_true = t.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    loss = loss_f(a, b)
    print(loss)
