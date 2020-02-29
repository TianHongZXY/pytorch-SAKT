import torch
import numpy as np
from torch.autograd import Variable

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(x, pad):
    "Create a mask to hide padding and future words."
    mask = torch.unsqueeze((x!=pad), -1)

    # tgt_mask是mask掉pad，sub_mask是mask掉future words
    #         print('tgt_mask size before: ', tgt_mask.size())
    tgt_mask = mask & Variable(
        subsequent_mask(x.size(-1)).type_as(mask.data))
    #         print('tgt_mask size after: ', tgt_mask.size())
    return tgt_mask