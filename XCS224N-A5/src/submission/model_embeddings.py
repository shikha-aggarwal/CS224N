#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from .cnn import CNN
from .highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### START CODE HERE for part 1f
        self.embed_size = embed_size
        self.kernel_size = 5
        self.dropout_prob = 0.3
        self.vocab = vocab
        self.embeddings = nn.Embedding(num_embeddings = len(vocab.char2id),
                                       embedding_dim = self.embed_size,
                                       padding_idx = vocab.char2id['<pad>'])


        self.cnn_layer = CNN(num_filters = self.embed_size,
                             kernel_size = (self.kernel_size, self.embed_size))

        self.highway_layer = Highway(embed_size = self.embed_size,
                                dropout_prob = self.dropout_prob)
        ### END CODE HERE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### START CODE HERE for part 1f
        embeddings_tensor = self.embeddings(input_tensor)
        embeddings_tensor_shape = embeddings_tensor.shape
        x = embeddings_tensor.view(embeddings_tensor_shape[0] * embeddings_tensor_shape[1],
                                   1,
                                   embeddings_tensor_shape[2],
                                   embeddings_tensor_shape[3])

        conv_x = self.cnn_layer(x)
        embeddings = self.highway_layer(conv_x)

        batched_emb = embeddings.view(input_tensor.shape[0], input_tensor.shape[1], self.embed_size)

        return batched_emb
        ### END CODE HERE

