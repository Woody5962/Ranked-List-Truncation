import torch as t
from torch import nn
import torch.nn.functional as F
import math

from .metrics import Metric_for_Loss


class BiCutLoss(nn.Module):
    """bicut对应的loss
    
    """
    def __init__(self, alpha: float=0.65, r: float=0.0971134020, metric: str='nci'):
        super(BiCutLoss, self).__init__()
        self.metric = metric
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
            mask[i][self.slice_index(output[i]) + 1:] = 0
        r = t.ones_like(output)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if self.metric == 'nci': 
                    r[i][j] = t.tensor([0, -1 / math.log2(j+2)]) if labels[i][j] == 1 else t.tensor([0, (j+1) / self.alpha])
                else: 
                    r[i][j] = t.tensor([(1 - self.alpha) / self.r, 0]) if labels[i][j] == 1 else t.tensor([0, self.alpha / (1 - self.r)])
        
        mask_output = output.mul(mask)
        loss_matrix = mask_output.mul(r)
        return t.sum(loss_matrix).div(output.shape[0])


class ChoopyLoss(nn.Module):
    """Choopy对应的loss

    """
    def __init__(self, metric: str='f1'):
        super(ChoopyLoss, self).__init__()
        self.metric = metric
    
    def forward(self, output, labels):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j+1)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j+1)
        
        loss_matrix = output.squeeze().mul(r)
        return -t.sum(loss_matrix).div(output.shape[0])


class AttnCutLoss(nn.Module):
    """AttnCut对应的loss

    """
    def __init__(self, metric: str='f1', tau: float=0.95):
        super(AttnCutLoss, self).__init__()
        self.metric = metric
        self.tau = tau
    
    def forward(self, output: t.Tensor, labels: t.Tensor):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j+1)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j+1)
        q = t.exp(r.div(self.tau))
        norm_factor = t.sum(q, axis=1).unsqueeze(dim=1)
        q = q.div(norm_factor)
        
        output = t.log(output.squeeze())
        loss_matrix = output.mul(q)
        return -t.sum(loss_matrix).div(output.shape[0])
    
    
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
        super(RerankLoss, self).__init__()
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
            if t.sum(sample_label) == 0: return t.tensor(0.)
            y_pos, y_neg = [], []
            n_pos, n_neg = 0, 0
            for i, label in enumerate(sample_label):
                if label: 
                    y_pos.append(sample_pred[i])
                    n_pos += 1
                else: 
                    y_neg.append(sample_pred[i])
                    n_neg += 1
            y_pos_1D = t.tensor(y_pos).unsqueeze(-1).expand(-1, n_neg).flatten().float()
            y_neg_1D = t.tensor(y_neg).repeat(n_pos).float()
            y_true = t.ones_like(y_pos_1D).float()
            loss.append(F.margin_ranking_loss(
                y_pos_1D, y_neg_1D, y_true,
                margin=self.margin,
                reduction=self.reduction
            ))
        return t.sum(t.tensor(loss)).div(output.shape[0])

        
class MtCutLoss(nn.Module):
    """MtCut的loss，尝试加入分类loss

    Args:
        nn ([type]): [description]
    """
    def __init__(self, metric: str='f1', rerank_weight: float=0.5, classi_weight: float=0.5, num_tasks: float=3):
        super(MtCutLoss, self).__init__()
        self.rerank_weight, self.classi_loss = rerank_weight, classi_weight
        self.weights = nn.Parameter(t.randn(int(num_tasks)), requires_grad=True)
        self.cutloss = AttnCutLoss(metric=metric)
        self.rerankloss = RerankLoss()
        self.classiloss = nn.BCELoss()
        self.num_tasks = num_tasks
        
    def forward(self, output, labels):
        if self.num_tasks == 3: pred_y, rerank_y, cut_y = output
        elif self.num_tasks == 2.1: pred_y, cut_y = output
        else: rerank_y, cut_y = output
        class_label = rerank_label = cut_label = labels
        cutloss = self.cutloss(cut_y, cut_label)
        if self.num_tasks == 3 or self.num_tasks == 2.2: rerankloss = self.rerankloss(rerank_y, rerank_label).mul(self.rerank_weight)
        if self.num_tasks == 3 or self.num_tasks == 2.1: classiloss = self.classiloss(pred_y.squeeze(), class_label).mul(self.classi_loss)
        # print('cutloss: {} | rerankloss: {} | classify_loss: {}'.format(cutloss, rerankloss, classiloss))
        if self.num_tasks == 3: return cutloss.add(rerankloss).add(classiloss)
        elif self.num_tasks == 2.1: return cutloss.add(classiloss)
        else: return cutloss.add(rerankloss)


class MtCutLoss1(nn.Module):
    """MtCut的loss，尝试加入分类loss

    Args:
        nn ([type]): [description]
    """
    def __init__(self, metric: str='f1', num_tasks: float=3):
        super(MtCutLoss1, self).__init__()
        self.weights = nn.Parameter(t.randn(int(num_tasks)), requires_grad=True)
        self.cutloss = AttnCutLoss(metric=metric)
        self.rerankloss = RerankLoss()
        self.classiloss = nn.BCELoss()
        self.num_tasks = num_tasks
        
    def forward(self, output, labels):
        if self.num_tasks == 3: pred_y, rerank_y, cut_y = output
        elif self.num_tasks == 2.1: pred_y, cut_y = output
        else: rerank_y, cut_y = output
        class_label = rerank_label = cut_label = labels
        cutloss = self.cutloss(cut_y, cut_label)
        if self.num_tasks == 3 or self.num_tasks == 2.2: rerankloss = self.rerankloss(rerank_y, rerank_label)
        if self.num_tasks == 3 or self.num_tasks == 2.1: classiloss = self.classiloss(pred_y.squeeze(), class_label)
        # print('cutloss: {} | rerankloss: {} | classify_loss: {}'.format(cutloss, rerankloss, classiloss))
        if self.num_tasks == 3: return t.stack((classiloss, rerankloss, cutloss)) @ self.weights
        elif self.num_tasks == 2.1: return t.stack((classiloss, cutloss)) @ self.weights
        else: return t.stack((rerankloss, cutloss)) @ self.weights

        
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
    # loss_f = RerankLoss()
    # y_pred = t.tensor([1., 1.2, 3.1, 0.1, 1.1, 4.2, 2.1, 5.1]).unsqueeze(dim=-1)
    # y_true = t.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    # loss = loss_f(a, b)
    # print(loss)