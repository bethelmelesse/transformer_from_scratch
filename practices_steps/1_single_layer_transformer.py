from transformers import BertTokenizer
import torch
import torch.nn as nn
import math

print()

context_1 ="My name is Bethel. I name to eat dirkosh."
context_2 = "Obama is the US president. Then Trump is the old US president."

contexts = [context_1, context_2]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
vocab_size = tokenizer.vocab_size

tokenized = tokenizer(contexts, padding=True)
# print(tokenized)

token_input_ids = torch.LongTensor(tokenized["input_ids"])
token_attention_masks = torch.LongTensor(tokenized["attention_mask"])

batch_size = len(contexts)
embedding_dim = 128
seq_length = len(token_input_ids[0])          # 16 after padding for this examples 
max_seq_length  = 512


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # task 1: Intitialize w for query, key and value
        self.w_query = torch.rand(embedding_dim, embedding_dim)        # shape of w = (128 * 256)
        self.w_key = torch.rand(embedding_dim, embedding_dim) 
        self.w_value = torch.rand(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x, token_attention_masks):                        # x = 2, 16, 128
        # task 2: get query, key, value
        query = torch.matmul(x, self.w_query)    # shape of q, k, v = 2 * 16 * 256
        key = torch.matmul(x, self.w_key)          
        value = torch.matmul(x, self.w_value)

        # task 3: multiply query by key
        product = torch.bmm(query, key.transpose(1,2))          # shape of product = 2 * 16 * 16

        # task 4: scale by root dk - dimensiion of key
        dk = key.size(dim=2)
        root_dk = math.sqrt(dk)
        scale = product / root_dk      # attention score            # shape = 2 * 16 * 16

        # task 4.1: 
        new_attention_mask = token_attention_masks.clone()
        new_attention_mask[new_attention_mask == 0] = -1000        # shape = 2 * 16
        new_attention_mask[new_attention_mask == 1] = 0
        new_attention_mask = new_attention_mask.unsqueeze(1)
        new_scale = scale + new_attention_mask

        # task 5: softmax
        attention_probability = self.softmax(scale)      # shape of attention prob = 2 * 16 * 16

        # task 6: multiply attention prob by value and sum
        attention_output = torch.bmm(attention_probability, value)

        return attention_output


class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mean = 0
        # self.variance = 0

    def forward(self, x):
        mean = torch.mean(x, dim=2)
        mean = torch.unsqueeze(mean, 2)
        variance= torch.var(x, dim=2)
        variance = torch.unsqueeze(variance, 2)
        layer_norm = (x - mean) / variance
        return layer_norm

class Positional_embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.posit_embed_init = torch.arange(0, seq_length)
        self.posit_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
    def forward(self): 
        positional_embeddings = self.posit_embedding(self.posit_embed_init).unsqueeze(0)
        return positional_embeddings       # shape = 1 * 16 * 128


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.posit_embed = Positional_embed()
        self.attention = Attention()
        self.linear_1 = nn.Linear(embedding_dim, embedding_dim) 
        self.layer_norm_1 = LayerNormalization()       
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.relu = nn.ReLU()
        self.linear_3 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.layer_norm_2 = LayerNormalization() 

    def forward(self, x, token_attention_masks):
        # Task 1: create embedding lookup table 
        token_embeddings = self.embedding(x)     # x = (batch_size, seq_length) & (batch_size, seq_length, embedding_dim)  (2, 16, 128)
        position_embeddings = self.posit_embed()
        token_embeddings = token_embeddings + position_embeddings

        # Task 2: apply attention
        attention_output = self.attention(token_embeddings, token_attention_masks)

        # Task 3: add linear layer
        linear_layer_1 = self.linear_1(attention_output)

        # Task 4: add skip conn.
        residual_output_1 = linear_layer_1 + token_embeddings         # shape = 2 * 16 * 128
        
        # Task 5: apply first layer norm 
        layer_norm_output_1 = self.layer_norm_1(residual_output_1)      # shape = 2 * 16 * 128

        # Task 6: apply second linear layer
        linear_layer_2 = self.linear_2(layer_norm_output_1)           # shape = 2 * 16 * 512

        # Task 7: Apply Relu
        relu_output = self.relu(linear_layer_2)                    # shape = 2 * 16 * 512

        # Task 8: apply third linear layer
        linear_layer_3 = self.linear_3(relu_output)                  # shape = 2 * 16 * 128

        # Task 9: add skip conn.
        residual_output_2 = linear_layer_3 + layer_norm_output_1        # shape = 2 * 16 * 128

        # Task 10: apply second layer norm 
        layer_norm_output_2 = self.layer_norm_2(residual_output_2)      # shape = 2 * 16 * 128

        return layer_norm_output_2 



my_model = Model()
my_model(token_input_ids, token_attention_masks)

print()