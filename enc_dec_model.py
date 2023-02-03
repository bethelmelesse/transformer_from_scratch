import torch
import torch.nn as nn

from config import *
from embeddings import Input_embedding
from transformer_layers import (Decoder_transformer_layer,
                                Encoder_transformer_layer)


class Model(nn.Module):
    def __init__(self, vocab_size_source, vocab_size_target, vocab_target):
        super().__init__()
        self.input_embed_source = Input_embedding(vocab_size_source, MAX_SEQ_LENGTH_SOURCE)
        self.input_embed_target = Input_embedding(vocab_size_target, MAX_SEQ_LENGTH_TARGET)
        self.lookup_table = nn.Embedding(vocab_size_target, EMBEDDING_DIM)
        # vocab_target_values = torch.LongTensor([i for i in vocab_target.values()])
        self.softmax = nn.Softmax(dim=2)
        self.vocab_target = vocab_target
        self.loss = nn.CrossEntropyLoss(reduction='none')

        self.encoder_layers = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            self.encoder_layers.append(Encoder_transformer_layer())

        self.decoder_layers = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            self.decoder_layers.append(Decoder_transformer_layer())

    def forward(self, token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target, is_training):
        input_embeddings_source = self.input_embed_source(token_input_ids_source)
        input_embeddings_target = self.input_embed_target(token_input_ids_target[:, :-1])
        token_attention_masks_target_without_end = token_attention_masks_target[:, :-1]

        for n in range(NUM_LAYERS):
            encoder_output = self.encoder_layers[n](input_embeddings_source, token_attention_masks_source, token_attention_masks_target_without_end)
            input_embeddings_source = encoder_output

        if is_training == True:
            for n in range(NUM_LAYERS):
                decoder_output = self.decoder_layers[n](input_embeddings_target, token_attention_masks_source,
                                                    token_attention_masks_target_without_end, encoder_output)
                input_embeddings_target = decoder_output

        else:
            for n in range(NUM_LAYERS):
                for j in range(input_embeddings_target.shape[1]):
                    decoder_output = self.decoder_layers[n](input_embeddings_target[:,:j,:], token_attention_masks_source, token_attention_masks_target_without_end[:,:j], encoder_output)
                    input_embeddings_target = decoder_output

        # print(decoder_output.shape)          # ([2, 18, 128])
        look_up_table = self.lookup_table.weight.transpose(0, 1)          # ([250002, 128]) ---> ([128, 250002])
        dot_product = torch.matmul(decoder_output, look_up_table)         # ([2, 18, 250002])

        ground_truth = token_input_ids_target[:, 1:].reshape(-1)
        if is_training == True:
            index_highest_prob = None
            loss = self.loss(dot_product.view(-1, dot_product.shape[2]), ground_truth)
            seq_len_target = token_input_ids_target[:, :-1].shape[1]
            loss = loss.view(-1, seq_len_target)
            loss = loss * token_attention_masks_target_without_end
            loss = torch.sum(loss)/torch.sum(token_attention_masks_target_without_end)

        else:
            loss = None
            softmax = self.softmax(dot_product)                                        # ([2, 18, 250002])
            index_highest_prob = torch.argmax(softmax, dim=2)

        return index_highest_prob, loss
