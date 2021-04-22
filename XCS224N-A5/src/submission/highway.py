#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

### START CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.utils

class Highway(nn.Module):
    def __init__(self, embed_size, dropout_prob=0.2):
        super(Highway, self).__init__()
        torch.manual_seed(0)
        self.dropout_prob = dropout_prob
        self.embed_size = embed_size

        self.proj = nn.Linear(in_features = self.embed_size, out_features = self.embed_size, bias = True)
        self.relu = nn.ReLU()
        self.gate = nn.Linear(in_features = self.embed_size, out_features = self.embed_size, bias = True)
        self.dropout = nn.Dropout(p = self.dropout_prob)


    def forward(self, x_conv_out):
        """
        @param x_conv_out (Tensor): a tensor of shape (batch_size, embed_size)

        @returns word_emb (Tensor): a variable/tensor of shape (batch_size, embed_size, ) representing the words.
        """
        x_proj = self.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.proj(x_conv_out))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul(1 - x_gate, x_conv_out)
        x_word_emb = self.dropout(x_highway)
        return x_word_emb

### END CODE HERE 

