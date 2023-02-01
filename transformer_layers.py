from config import *
import torch
import torch.nn as nn
from sublayers import Attention_layer, LayerNormalization

class Encoder_transformer_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention_layer()                                           # step 1 - self-attention
        self.linear_1 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)                     # step 2 - linear
        self.layer_norm_1 = LayerNormalization()                                    # step 3 - layer norm
        self.linear_2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * 4)                    # step 4 - linear
        self.relu = nn.ReLU()                                                       # step 5 - relu
        self.linear_3 = nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM)                 # step 6 - linear
        self.layer_norm_2 = LayerNormalization()                                    # step 7 - layer norm

    def forward(self, input_embeddings_source, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding=None):
        
        attention_output = self.attention(input_embeddings_source, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding, attention_type='encoder')
    
        linear_layer_1 = self.linear_1(attention_output)
        residual_output_1 = linear_layer_1 + input_embeddings_source         # shape = 2 * 16 * 128

        layer_norm_output_1 = self.layer_norm_1(residual_output_1)      # shape = 2 * 16 * 128

        linear_layer_2 = self.linear_2(layer_norm_output_1)      # shape = 2 * 16 * 512
        relu_output = self.relu(linear_layer_2)                    # shape = 2 * 16 * 512

        linear_layer_3 = self.linear_3(relu_output)                  # shape = 2 * 16 * 128
        residual_output_2 = linear_layer_3 + layer_norm_output_1        # shape = 2 * 16 * 128
        layer_norm_output_2 = self.layer_norm_2(residual_output_2)      # shape = 2 * 16 * 128

        return layer_norm_output_2 

class Decoder_transformer_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_self_attention = Attention_layer()                               # step 1 - self-attention - masked self attention ---- make it masked
        self.linear_1 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)                       # step 2 - linear

        self.cross_attention = Attention_layer()                                     # step 3 - self-attention - cross attention --- make it masked
        self.linear_2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)                       # step 4 - linear

        self.layer_norm = LayerNormalization()                                      # step 5 - layer norm
        self.linear_3 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)                    # step 6 - linear
        self.relu = nn.ReLU()                                                       # step 7 - relu
        self.linear_4 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)                    # step 8 - linear

    def forward(self, input_embeddings_target, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding):
        
        masked_self_attention_output = self.masked_self_attention(input_embeddings_target, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding=None, attention_type='decoder')                            # decoder = 2 * 18 * 128
        linear_layer_1 = self.linear_1(masked_self_attention_output)         # decoder = 2 * 18 * 128
        residual_output_1 = linear_layer_1 + input_embeddings_target           # decoder = 2 * 18 * 128

        cross_attention_output = self.cross_attention(residual_output_1, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding=encoder_output_embedding, attention_type='cross')   # decoder = 2 * 18 * 128
        linear_layer_2 = self.linear_2(cross_attention_output)           # decoder = 2 * 18 * 128
        residual_output_2 = linear_layer_2 + residual_output_1            # decoder = 2 * 18 * 128

        layer_norm_output = self.layer_norm(residual_output_2)            # decoder = 2 * 18 * 128 
        linear_layer_3 = self.linear_3(layer_norm_output)               # decoder = 2 * 18 * 128
        relu_output = self.relu(linear_layer_3)                           # decoder = 2 * 18 * 128
        linear_layer_4 = self.linear_4(relu_output)                       # shape = 2 * 18 * 128
        residual_output_3 = linear_layer_4 + residual_output_2            # shape = 2 * 18 * 128

        return residual_output_3

