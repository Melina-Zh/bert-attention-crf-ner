# coding=utf-8
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from model import CRF
from model.models import Attention
from torch.autograd import Variable
import torch
import torch.nn.functional as F

import ipdb




class BERT_ATTENTION_CRF(nn.Module):
    """
    bert_attention_crf model
    """
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, d_model, d_k, d_v, dropout_ratio, dropout1, use_cuda=False):
        super(BERT_ATTENTION_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = BertModel.from_pretrained(bert_config)
        self.W = nn.Parameter(torch.zeros(size=(d_model, d_model)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attn_layer = Attention(d_model, dropout=dropout_ratio)
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.liner = nn.Linear(hidden_dim*2, tagset_size+2)
        self.tagset_size = tagset_size


    def forward(self, sentence, domain, attention_mask=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds, _ = self.word_embeds(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)

        attention_out = self.Attention(domain, embeds, embeds)
        hidden = sentence+F.relu(torch.bmm(attention_out, self.W)+torch.bmm(sentence, self.W))

        out = self.dropout1(hidden)

        return out

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value



