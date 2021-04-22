#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

### START CODE HERE for part 1e
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_filters, kernel_size):
        """
        :param num_filters (int): number of output channels
        :param kernel_size (int): window size of each filter
        """
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        self.conv_layer = nn.Conv1d(in_channels = 1,
                                    out_channels = self.num_filters,
                                    kernel_size = self.kernel_size,
                                    stride = 1)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size = 1)
        self.relu_layer = nn.ReLU()

    def forward(self, embeddings):
        """
        :param embeddings: char embeddings (batch_size, max_word_len)
        :return:
        """
        x_conv = self.conv_layer(embeddings)
        x_conv_shape = x_conv.shape
        ## collapse the last dimension of size 1
        x_conv_reshape = x_conv.view(x_conv_shape[0], x_conv_shape[1], x_conv_shape[2])

        x_conv_out = self.max_pool(self.relu_layer(x_conv_reshape))
        x_conv_out_shape = x_conv_out.shape

        x_conv_out_reshape = x_conv_out.view(x_conv_out_shape[0], x_conv_out_shape[1])

        return x_conv_out_reshape

### END CODE HERE

