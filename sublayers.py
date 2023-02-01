import math

import torch
import torch.nn as nn

from config import *


class Attention_layer(nn.Module):
    def __init__(self):
        super().__init__()
        # task 1: Intitialize w for query, key and value
        self.w_query = nn.Parameter(torch.rand(EMBEDDING_DIM, EMBEDDING_DIM))       # shape of w = (128 * 128)
        self.w_query = torch.nn.init.xavier_normal_(self.w_query)
        self.w_key = nn.Parameter(torch.rand(EMBEDDING_DIM, EMBEDDING_DIM))
        self.w_key = torch.nn.init.xavier_normal_(self.w_key)
        self.w_value = nn.Parameter(torch.rand(EMBEDDING_DIM, EMBEDDING_DIM))
        self.w_value = torch.nn.init.xavier_normal_(self.w_value)

        self.num_heads = 8
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_embeddings, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding, attention_type):      # x = 2, 16, 128
        batch_size = input_embeddings.shape[0]
        seq_length_source = len(token_attention_masks_source[0])
        seq_length_target = len(token_attention_masks_target[0])

        num_heads = self.num_heads
        head_dim = int(EMBEDDING_DIM / self.num_heads)          # 16

        def query(embeddings, seq_length):
            query = torch.matmul(embeddings, self.w_query)    # shape of q, k, v = 2 * 16 * 128         cross = 2, 18, 128
            # encoder = 2, 16, 16, 8   decoder = 2, 18, 16, 8    cross = decoder
            new_query = torch.reshape(query, (batch_size, seq_length, head_dim, num_heads))
            new_query = torch.transpose(new_query, 2, 3)                                         # encoder = 2, 16, 8, 16   decoder = 2, 18, 8, 16
            new_query = torch.transpose(new_query, 1, 2)                                         # encoder = 2, 8, 16, 16   decoder = 2, 8, 18, 16
            new_query = torch.reshape(new_query, (batch_size * num_heads, seq_length, head_dim))  # encoder = 16, 16, 16    decoder = 16, 18, 16
            return new_query

        def key(embeddings, seq_length):
            key = torch.matmul(embeddings, self.w_key)
            # encoder = 2, 16, 16, 8   decoder = 2, 18, 16, 8    cross = encoder
            new_key = torch.reshape(key, (batch_size, seq_length, head_dim, num_heads))
            new_key = torch.transpose(new_key, 2, 3)                                               # encoder = 2, 16, 8, 16   decoder = 2, 18, 8, 16
            new_key = torch.transpose(new_key, 1, 2)                                               # encoder = 2, 8, 16, 16   decoder = 2, 8, 18, 16
            new_key = torch.reshape(new_key, (batch_size * num_heads, seq_length, head_dim))        # encoder = 16, 16, 16     decoder = 16, 18, 16
            return new_key

        def value(embeddings, seq_length):
            value = torch.matmul(embeddings, self.w_value)
            # encoder = 2, 16, 16, 8   decoder = 2, 18, 16, 8  cross = encoder
            new_value = torch.reshape(value, (batch_size, seq_length, head_dim, num_heads))
            new_value = torch.transpose(new_value, 2, 3)                                          # encoder = 2, 16, 8, 16   decoder = 2, 18, 8, 16
            new_value = torch.transpose(new_value, 1, 2)                                          # encoder = 2, 8, 16, 16   decoder = 2, 8, 18, 16
            new_value = torch.reshape(new_value, (batch_size * num_heads, seq_length, head_dim))   # encoder = 16, 16, 16     decoder = 16, 18, 16
            return new_value

        def scaled_product(query, key):
            # encoder = 16, 16, 16        decoder = 16, 18, 18    cross = 16, 18, 16
            product_across_heads = torch.bmm(query, key.transpose(1, 2))
            d_head_dim = key.size(dim=2)                                  # encoder = 16        decoder = 16       cross = 16
            root_d_head_dim = math.sqrt(d_head_dim)                            # encoder = 4        decoder = 4        cross = 4
            scale = product_across_heads / root_d_head_dim      # attention score    # encoder = 16, 16, 16       decoder = 16, 18, 18   # cross = 16, 18, 16
            return scale

        def attention_mask(attention_mask, seq_length):
            attention_mask[attention_mask == 0] = -1000                                            # encoder = 2 * 16            # cross = 2 * 18
            attention_mask[attention_mask == 1] = 0                                                # encoder = 2 * 16            # cross = 2 * 18
            attention_mask = attention_mask.unsqueeze(1)                                           # encoder = 2 * 1 * 16        # cross = 2 * 1 * 18
            attention_mask = attention_mask.repeat_interleave(repeats=num_heads, dim=1)            # encoder = 2 * 8 * 16        # cross = 2 * 8 * 18
            attention_mask = attention_mask.reshape(batch_size * num_heads, seq_length)            # encoder = 16 * 16           # cross = 16 * 18
            return attention_mask

        def decoder_attention_mask():
            decoder_attention_mask_1 = token_attention_masks_target.clone()                                    # cross = 2 * 18
            decoder_attention_mask_1 = decoder_attention_mask_1.unsqueeze(1)                                           # shape = 2 * 1 * 18
            decoder_attention_mask_1 = decoder_attention_mask_1.repeat_interleave(repeats=seq_length, dim=1)           # shape = 2 * 18 * 18

            decoder_attention_mask_2 = torch.ones_like(decoder_attention_mask_1) * (-1000)
            decoder_attention_mask_2 = torch.triu(decoder_attention_mask_2, diagonal=1)                           # masked = 2, 18, 18

            decoder_attention_mask = decoder_attention_mask_2.repeat_interleave(repeats=num_heads, dim=0)           # shape = 16 * 18 * 18
            return decoder_attention_mask

        def softmax_sum_prob(attention_score, seq_length, value):
            # encoder = 16, 16, 16       decoder = 16, 18, 18    cross = 16, 18, 16
            attention_probabilty_across_heads = self.softmax(attention_score)
            # encoder = 16, 16, 16    decoder = 16, 18, 18    cross = 16, 18, 16
            prob_by_value_and_sum = torch.bmm(attention_probabilty_across_heads, value)
            prob_by_value_and_sum = torch.reshape(prob_by_value_and_sum, (batch_size, num_heads, seq_length, head_dim))
            # encoder = 2, 16, 8, 16    decoder = 2, 18, 8, 16   cross = 2, 18, 8, 16
            attention_output = torch.transpose(prob_by_value_and_sum, 1, 2)
            # encoder = 2, 16, 16, 8    decoder = 2, 18, 16, 8   cross = 2, 18, 16, 8
            attention_output = torch.transpose(attention_output, 2, 3)
            attention_output = torch.reshape(attention_output, (batch_size, seq_length, head_dim * num_heads)
                                             )    # encoder = 2, 16, 128   decoder =  2, 18, 128  cross = 2, 18, 128
            return attention_output

        if (attention_type == 'encoder'):
            seq_length = seq_length_source
            encoder_query = query(input_embeddings, seq_length)
            encoder_key = key(input_embeddings, seq_length)
            encoder_value = value(input_embeddings, seq_length)

            scaled_product = scaled_product(encoder_query, encoder_key)

            encoder_attention_mask = token_attention_masks_source.clone()
            attention_score = scaled_product + attention_mask(encoder_attention_mask, seq_length).unsqueeze(-1)            # encoder = 16, 16, 16
            attention_output = softmax_sum_prob(attention_score, seq_length, encoder_value)

        elif (attention_type == 'decoder'):
            seq_length = seq_length_target
            decoder_query = query(input_embeddings, seq_length)
            decoder_key = key(input_embeddings, seq_length)
            decoder_value = value(input_embeddings, seq_length)

            scaled_product = scaled_product(decoder_query, decoder_key)

            attention_score = scaled_product + decoder_attention_mask()                                   # decoder = 16, 18, 18
            attention_output = softmax_sum_prob(attention_score, seq_length, decoder_value)

        elif (attention_type == 'cross'):
            seq_length_source = seq_length_source
            seq_length_target = seq_length_target

            cross_query = query(input_embeddings, seq_length_target)
            cross_key = key(encoder_output_embedding, seq_length_source)
            cross_value = value(encoder_output_embedding, seq_length_source)

            scaled_product = scaled_product(cross_query, cross_key)

            cross_attention_mask = token_attention_masks_target.clone()                                       # cross = 2 * 18
            cross_attention_mask = attention_mask(cross_attention_mask, seq_length_target)
            cross_attention_mask = cross_attention_mask.unsqueeze(2)                                          # cross = 16 * 18 * 1
            attention_score = scaled_product + cross_attention_mask                                         # cross = 16, 16, 16
            attention_output = softmax_sum_prob(attention_score, seq_length_target, cross_value)

        return attention_output


class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=2)
        mean = torch.unsqueeze(mean, 2)
        variance = torch.var(x, dim=2)
        variance = torch.unsqueeze(variance, 2)
        layer_norm = (x - mean) / variance
        return layer_norm
