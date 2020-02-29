import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import copy
from torch.nn import LayerNorm


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, key_masks=None, query_masks=None, future_masks=None, dropout=None, infer=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    layernorm = LayerNorm(d_k).cuda()
    # query shape = [nbatches, h, T_q, d_k]       key shape = [nbatches, h, T_k, d_k] == value shape
    # scores shape = [nbatches, h, T_q, T_k]  == p_attn shape
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # if key_masks is not None:
    #     scores = scores.masked_fill(key_masks.unsqueeze(1).cuda() == 0, -1e9)
    if future_masks is not None:
        scores = scores.masked_fill(future_masks.unsqueeze(0).cuda() == 0, -1e9)


    p_attn = F.softmax(scores, dim=-1)
    outputs = p_attn
    # if query_masks is not None:
    #     outputs = outputs * query_masks.unsqueeze(1)
    if dropout is not None:
        outputs = dropout(outputs)
    outputs = torch.matmul(outputs, value)

    outputs += query
    return layernorm(outputs), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.2, infer=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = LayerNorm(d_model).cuda()
        self.infer = infer

    def forward(self, query, key, value, key_masks=None, query_masks=None, future_masks=None):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [F.relu(l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2), inplace=True)
             for l, x in zip(self.linears, (query, key, value))]
        # k v shape = [nbatches, h, T_k, d_k],  d_k * h = d_model
        # q shape = [nbatches, h, T_q, d_k]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, query_masks=query_masks,
                                 key_masks=key_masks, future_masks=future_masks, dropout=self.dropout, infer=self.infer)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.layernorm(x)