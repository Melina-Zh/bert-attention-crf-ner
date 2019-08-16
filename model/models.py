import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, d_model, attn_dropout=0.1):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias = False)
        self.U = nn.Linear(d_model, d_model, bias = False)
        self.d_model=d_model
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, q, k, v, mask=None):
        #print(k.shape)
        #print(q.shape)
        batch_size = k.size(0)
        seq_len = k.size(1)
        q = q.view(batch_size, 1, self.d_model)
        q = q.expand(batch_size, seq_len, self.d_model)
        W = self.W.weight.unsqueeze(0).expand(batch_size, self.d_model, self.d_model)
        U = self.U.weight.unsqueeze(0).expand(batch_size, self.d_model, self.d_model)
        fac1 = torch.bmm(k, W)
        fac2 = torch.bmm(q, U)

        s = torch.bmm(v.transpose(1, 2), self.tanh(fac1 + fac2))

        attn = self.softmax(s)
        attn = self.dropout(attn)

        output = torch.bmm(v, attn)

        return output
