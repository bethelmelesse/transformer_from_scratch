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
        self.w_query = torch.rand(embedding_dim, embedding_dim)        # shape of w = (128 * 128)
        self.w_key = torch.rand(embedding_dim, embedding_dim) 
        self.w_value = torch.rand(embedding_dim, embedding_dim)

        self.num_heads = 8
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, input_embeddings, token_attention_masks):                        # x = 2, 16, 128
        # task 2: get query, key, value
        query = torch.matmul(input_embeddings, self.w_query)    # shape of q, k, v = 2 * 16 * 128
        key = torch.matmul(input_embeddings, self.w_key)          
        value = torch.matmul(input_embeddings, self.w_value)

        # task 3: split query, key and value into attention heads and reshape 
        head_dim = int(embedding_dim / self.num_heads)          # 16

        # task 3.1: for query
        # Reshape to Batch, seq_length, head_dim, num_head
        new_query = torch.reshape(query, (batch_size, seq_length, head_dim, self.num_heads))      # shape = 2, 16, 16, 8
        # swap -->  Batch, seq_length, num_head, head_dim
        new_query = torch.transpose(new_query, 2, 3)                  # shape = 2, 16, 8, 16
        # swap -->  Batch, num_head, seq_length, head_dim
        new_query = torch.transpose(new_query, 1, 2)                     # shape = 2, 8, 16, 16
        # reshape again --->
        new_query = torch.reshape(new_query, (batch_size * num_heads, seq_length, head_dim))         # shape = 16, 16, 16

         # task 3.2: for key
        new_key = torch.reshape(key, (batch_size, seq_length, head_dim, self.num_heads))             # shape = 2, 16, 16, 8
        new_key = torch.transpose(new_key, 2, 3)                                                     # shape = 2, 16, 8, 16
        new_key = torch.transpose(new_key, 1, 2)                                                     # shape = 2, 8, 16, 16
        new_key = torch.reshape(new_key, (batch_size * num_heads, seq_length,head_dim))             

         # task 3.3: for value
        new_value = torch.reshape(value, (batch_size, seq_length, head_dim, self.num_heads))
        new_value = torch.transpose(new_value, 2, 3)                                                 # shape = 2, 16, 8, 16
        new_value = torch.transpose(new_value, 1, 2)                                                 # shape = 2, 8, 16, 16 
        new_value = torch.reshape(new_value, (batch_size * num_heads, seq_length, head_dim))         # shape = 16, 16, 16

        # task 4: multiply query by key
        product_across_heads = torch.bmm(new_query, new_key.transpose(1, 2))                        # shape = 16, 16, 16

        # task 5: scale the product
        d_head_dim = new_key.size(dim=2)
        root_d_head_dim = math.sqrt(d_head_dim)
        scale = product_across_heads / root_d_head_dim      # attention score            # shape = 16, 16, 16

        # task 6: include attention mask  
        new_attention_mask = token_attention_masks.clone()
        new_attention_mask[new_attention_mask == 0] = -1000                                            # shape = 2 * 16
        new_attention_mask[new_attention_mask == 1] = 0                                                # shape = 2 * 16
        new_attention_mask = new_attention_mask.unsqueeze(1)                                           # shape = 2 * 1 * 16
        new_attention_mask = new_attention_mask.repeat_interleave(repeats=num_heads, dim=1)           # shape = 2 * 8 * 16
        new_attention_mask = new_attention_mask.reshape(batch_size * num_heads, seq_length)           # shape = 16 * 16

        attention_score = scale + new_attention_mask                                                  # shape = 16, 16, 16

        # task 7: softmax the attention score 
        attention_probabilty_across_heads = self.softmax(attention_score)                             # shape = 16, 16, 16        
        
        # task 8: multiply attention prob by value and sum
        prob_by_value_and_sum = torch.bmm(attention_probabilty_across_heads, new_value)                                       # shape = 16, 16, 16
        prob_by_value_and_sum = torch.reshape(prob_by_value_and_sum, (batch_size, num_heads, seq_length, head_dim))           # shape = 2, 8, 16, 16

        # task 9: reorder 
        attention_output = torch.transpose(prob_by_value_and_sum , 1, 2)                                             # shape = 2, 16, 8, 16
        attention_output = torch.transpose(attention_output, 2, 3)                                                   # shape = 2, 16, 16, 8
        attention_output = torch.reshape(attention_output, (batch_size, seq_length, head_dim * num_heads))           # shape = 2, 16, 128
      
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

class Input_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.posit_embed = Positional_embed()

    def forward(self, token_input_ids):
        token_embeddings = self.embedding(token_input_ids)     # x = (batch_size, seq_length) & (batch_size, seq_length, embedding_dim)  (2, 16, 128)
        position_embeddings = self.posit_embed()
        token_embeddings_with_posit = token_embeddings + position_embeddings

        return token_embeddings_with_posit

class Transformer_layer(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.attention = Attention()
        self.linear_1 = nn.Linear(embedding_dim, embedding_dim) 
        self.layer_norm_1 = LayerNormalization()       
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.relu = nn.ReLU()
        self.linear_3 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.layer_norm_2 = LayerNormalization()

    def forward(self, input_embeddings, token_attention_masks):
        attention_output = self.attention(input_embeddings, token_attention_masks)
        linear_layer_1 = self.linear_1(attention_output)
        residual_output_1 = linear_layer_1 + input_embeddings         # shape = 2 * 16 * 128
        layer_norm_output_1 = self.layer_norm_1(residual_output_1)      # shape = 2 * 16 * 128
        linear_layer_2 = self.linear_2(layer_norm_output_1)           # shape = 2 * 16 * 512
        relu_output = self.relu(linear_layer_2)                    # shape = 2 * 16 * 512
        linear_layer_3 = self.linear_3(relu_output)                  # shape = 2 * 16 * 128
        residual_output_2 = linear_layer_3 + layer_norm_output_1        # shape = 2 * 16 * 128
        layer_norm_output_2 = self.layer_norm_2(residual_output_2)      # shape = 2 * 16 * 128

        return layer_norm_output_2 


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embed= Input_embedding()
        self.transformer_layer = Transformer_layer(num_heads)
         
    def forward(self, token_input_ids, token_attention_masks):
        input_embeddings = self.input_embed(token_input_ids)
        transformer_layer = self.transformer_layer(input_embeddings, token_attention_masks)

        return transformer_layer


num_heads = 8
my_model = Model()
a = my_model(token_input_ids, token_attention_masks)

print(a)
print(a.shape)



print()