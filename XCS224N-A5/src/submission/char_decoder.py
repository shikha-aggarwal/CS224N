#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        ### START CODE HERE for part 2a
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.vocab_size = len(target_vocab.char2id)

        self.charDecoder = nn.LSTM(input_size = char_embedding_size,
                                   hidden_size = hidden_size)

        self.char_output_projection = nn.Linear(in_features = hidden_size,
                                                out_features = self.vocab_size,
                                                bias = True)

        self.decoderCharEmb = nn.Embedding(num_embeddings = self.vocab_size,
                                           embedding_dim = char_embedding_size,
                                           padding_idx = target_vocab.char2id['<pad>'])
        ### END CODE HERE()

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### TODO - Implement the forward pass of the character decoder.
        ### START CODE HERE for part 2b
        embeddings_tensor = self.decoderCharEmb(input) ## shape: length, batch, char_embedding_size
        lstm_output, lstm_hidden = self.charDecoder(embeddings_tensor, dec_hidden) ## shape: length, batch, hidden size
        score = self.char_output_projection(lstm_output) ## shape = length, batch, len(target_vocab.char2id)
        return score, lstm_hidden
        ### END CODE HERE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        ### START CODE HERE for part 2c
        input_sequence = char_sequence[:char_sequence.shape[0]-1, :]
        output_sequence = char_sequence[1:, :]
        score, hidden_state = self.forward(input = input_sequence,
                                           dec_hidden = dec_hidden)
        score_shape = score.shape
        score_reshape = score.view(score_shape[0] * score_shape[1], score_shape[2])
        output_sequence_shape = output_sequence.shape
        output_sequence_reshape = output_sequence.view(output_sequence_shape[0] * output_sequence_shape[1])
        loss = nn.CrossEntropyLoss(ignore_index = self.target_vocab.char2id['<pad>'], reduction = 'sum')
        loss_sum = loss(input = score_reshape, target = output_sequence_reshape)

        return loss_sum
        ### END CODE HERE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        ### START CODE HERE for part 2d

        #tensor of integers, shape (length, batch)
        batch_size = initialStates[0].shape[1]
        dec_hidden = initialStates

        current_chars = torch.tensor([self.target_vocab.start_of_word] * batch_size, device = device) # shape: (batch, )

        decoded_words = [''] * batch_size

        for iter_count in range(max_length):
            score, dec_hidden = self.forward(input = current_chars.view(1, batch_size),
                                             dec_hidden = dec_hidden)
            current_chars = torch.argmax(score, dim = -1).squeeze(0) ## shape: (batch, )
            batch_next_chars = [self.target_vocab.id2char.get(index, '<unk>') for index in current_chars.tolist()]
            decoded_words = [i + j for i, j in zip(decoded_words, batch_next_chars)]

        for idx in range(len(decoded_words)):
            decoded_word = decoded_words[idx]   #remove start_of_word
            end_word_at = max_length
            if '}' in decoded_word:
                end_word_at = decoded_word.index('}')
            decoded_words[idx] = decoded_word[:end_word_at]
        return decoded_words
        ### END CODE HERE
