'''
swish.py
Usage: similar to torch.nn.ReLU()...and torch.autograd.Function
# Based on: Ceshine Lee
https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
# Author: Ty Nguyen
# Contact: tynguyen.tech@gmail.com
'''

import torch
import torch.nn as nn


class Swish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor):
        return Swish_func.apply(input_tensor)