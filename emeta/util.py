# +
import torch


def kldiv(p, q):
    """returns p*log(p/q)"""
    return p*torch.where(q > torch.finfo().eps,
                         (p/q).log(), torch.zeros(1))
