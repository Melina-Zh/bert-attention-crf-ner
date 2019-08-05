import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, d_model, attn_dropout=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(size=(d_model, d_model)))
        self.U = nn.Parameter(torch.zeros(size=(d_model, d_model)))
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, q, k, v, mask=None):
        fac1 = torch.bmm(k, self.W)
        fac2 = torch.bmm(q, self.U)

        s = torch.bmm(v.transpose(1, 2), self.tanh(fac1 + fac2))

        attn = self.softmax(s)
        attn = self.dropout(attn)

        output= torch.bmm(v, attn)

        return output
