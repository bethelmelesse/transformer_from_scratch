import torch.nn as nn

from config import *


class Positional_embed(nn.Module):
    def __init__(self, max_seq_length):
        super().__init__()
        self.posit_embedding = nn.Embedding(max_seq_length, EMBEDDING_DIM)

    def forward(self, seq_length):
        posit_embed_init = torch.arange(0, seq_length).to(device=device)
        positional_embeddings = self.posit_embedding(posit_embed_init).unsqueeze(0)
        return positional_embeddings       # shape = 1 * 16 * 128


class Input_embedding(nn.Module):
    def __init__(self, vocab_size, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.posit_embed = Positional_embed(max_seq_length)

    def forward(self, token_input_ids):
        # x = (batch_size, seq_length_source) & (batch_size, seq_length_source, embedding_dim)  (2, 16, 128)
        token_embeddings = self.embedding(token_input_ids)
        seq_length = len(token_input_ids[0])
        position_embeddings = self.posit_embed(seq_length)
        token_embeddings_with_posit = token_embeddings + position_embeddings

        return token_embeddings_with_posit
