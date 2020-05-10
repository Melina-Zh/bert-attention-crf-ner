# coding=utf-8
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from model import CRF
from model.models import Attention
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import ipdb


class BERT_CRF(nn.Module):
    """
    bert-crf model
    """
    def __init__(self, bert_config, tagset_size, embedding_dim, d_model, dropout_ratio, dropout1, use_cuda=False):
        super(BERT_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.bert = BertModel.from_pretrained(bert_config)
        self.tagset_size = tagset_size
        self.d_model = d_model
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(tagset_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.liner = nn.Linear(d_model, tagset_size)


    def forward(self, input_ids, attention_mask=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''


        embeds, _ = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)

        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)

        #print("domain_shape:")
        #print(torch.Tensor(domain_embeds).shape)

        #print("hidden"+str(hidden.shape))
        hidden_out=embeds.contiguous().view(-1, self.d_model)
        #print("hiddenout" + str(hidden_out.shape))
        out = self.dropout1(hidden_out)
        #print("before liner"+str(out.shape))
        feats = self.liner(out)
        #print("after liner" + str(feats.shape))
        feats_out = feats.contiguous().view(batch_size, seq_length, -1)

        return feats_out

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        #print(feats.shape)
        #print(mask.shape)
        #print(tags.shape)
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        return loss_value



