#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_sum, scatter_min


def variadic_arange(size):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_arange

    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range


def variadic_meshgrid(input1, size1, input2, size2):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_meshgrid
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


def scatter_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    '''
    from https://github.com/rusty1s/pytorch_scatter/issues/48
    WARN: the range between src.max() and src.min() should not be too wide for numerical stability

    reproducible
    '''
    # f_src = src.float()
    # f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    # norm = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    # perm = norm.argsort(dim=dim, descending=descending)

    # return src[perm], perm
    src, src_perm = torch.sort(src, dim=dim, descending=descending)
    index = index.take_along_dim(src_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    src = src.take_along_dim(index_perm, dim=dim)
    perm = src_perm.take_along_dim(index_perm, dim=0)
    return src, perm


def scatter_topk(src: torch.Tensor, index: torch.Tensor, k: int, dim=0, largest=True):
    indices = torch.arange(src.shape[dim], device=src.device)
    src, perm = scatter_sort(src, index, dim, descending=largest)
    index, indices = index[perm], indices[perm]
    mask = torch.ones_like(index).bool()
    mask[k:] = index[k:] != index[:-k]
    return src[mask], indices[mask]


def fully_connect_edges(batch_ids):
    lengths = scatter_sum(torch.ones_like(batch_ids), batch_ids, dim=0)
    row, col = variadic_meshgrid(
        input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        size1=lengths,
        input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        size2=lengths,
    )
    return torch.stack([row, col], dim=0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return (self.avg).item()


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)


def load_checkpoint(model_path):
    return torch.load(model_path)


def stable_norm(input, *args, **kwargs):
    '''
        For L2: norm = sqrt(\sum x^2) = (\sum x^2)^{1/2}
        The gradient will have zero in divider if \sum x^2 = 0
        It is not ok to direct add eps to all x, since x might
        be a small but negative value.
        This function deals with this problem
    '''
    input = input.clone()
    with torch.no_grad():
        sign = torch.sign(input)
        input = torch.abs(input)
        input.clamp_(min=1e-10)
        input = sign * input
    return torch.norm(input, *args, **kwargs)
