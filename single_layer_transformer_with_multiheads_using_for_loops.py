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

class Multi_heads(nn.Module):
    def __init__(self):
        super().__init__()
 

    def forward(input_embeddings):
        

        return 0


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # task 1: Intitialize w for query, key and value
        self.w_query = torch.rand(embedding_dim, embedding_dim)        # shape of w = (128 * 128)
        self.w_key = torch.rand(embedding_dim, embedding_dim) 
        self.w_value = torch.rand(embedding_dim, embedding_dim)

        self.num_heads = 8

        self.softmax = nn.Softmax(dim=2)

        self.linear_after_conc = nn.Linear(embedding_dim, embedding_dim)

    
    def forward(self, input_embeddings, token_attention_masks):                        # x = 2, 16, 128
        # task 2: get query, key, value
        
        query = torch.matmul(input_embeddings, self.w_query)    # shape of q, k, v = 2 * 16 * 128
        key = torch.matmul(input_embeddings, self.w_key)          
        value = torch.matmul(input_embeddings, self.w_value)
        
        # task 2.1: split data across attention heads
        head_dim = int(embedding_dim / self.num_heads)          # 16
        new_query = []
        start_dim = 0
        head_dim_constant = head_dim
        for i in range(self.num_heads):                              # shape = 2 * 16 * 16
            new_query.append(query[:, :, start_dim : head_dim]) 
            start_dim = head_dim
            head_dim += head_dim_constant

        new_key = []
        head_dim = int(embedding_dim / self.num_heads)          # 16     
        head_dim_constant = head_dim
        start_dim = 0
        for i in range(self.num_heads):
            new_key.append(key[:, :, start_dim : head_dim])     # shape = 2 * 16 * 16
            start_dim = head_dim
            head_dim += head_dim_constant
        
        new_value = []    
        head_dim = int(embedding_dim / self.num_heads)          # 16 
        head_dim_constant = head_dim
        start_dim = 0
        for i in range(self.num_heads):
            new_value.append(value[:, :, start_dim : head_dim])     # shape = 2 * 16 * 16
            start_dim = head_dim
            head_dim += head_dim_constant
            
            
        # task 3: multiply query by key
        # product = torch.bmm(query, key.transpose(1,2))          # shape of product = 2 * 16 * 16
        product_across_heads =  []
        scale_heads = []
        for i in range(num_heads):
            product_across_heads.append(torch.bmm(new_query[i], new_key[i].transpose(1,2)))

            # task 4: scale by root dk - dimensiion of key
            # dk = key.size(dim=2)
            # root_dk = math.sqrt(dk)
            # scale = product / root_dk      # attention score            # shape = 2 * 16 * 16
            dk = key.size(dim=2)
            root_dk = math.sqrt(dk)
            scale = product_across_heads[i] / root_dk      # attention score            # shape = 2 * 16 * 16
            scale_heads.append(scale)

        # task 4.1: 
        new_attention_mask = token_attention_masks.clone()
        new_attention_mask[new_attention_mask == 0] = -1000        # shape = 2 * 16
        new_attention_mask[new_attention_mask == 1] = 0
        new_attention_mask = new_attention_mask.unsqueeze(1)

        # new_scale = scale + new_attention_mask
        new_scale_heads = []
        attention_probability_heads = []
        attention_output_heads = []
        for i in range(num_heads):
            new_scale_heads.append(scale_heads[i] + new_attention_mask)

            # task 5: softmax
            attention_probability_heads.append(self.softmax(new_scale_heads[i]))       # shape of attention prob = 2 * 16 * 16

            # task 6: multiply attention prob by value and sum
            product_heads = torch.bmm(attention_probability_heads[i], new_value[i])
            attention_output_heads.append(product_heads)

        attention_output = torch.concat((attention_output_heads[0], attention_output_heads[1]), -1)
        for i in range(num_heads - 2):
            attention_output = torch.concat((attention_output, attention_output_heads[1 + i]), -1)                        # shape = ([18, 16, 16])
           
        final_attention_output = self.linear_after_conc(attention_output)

        return final_attention_output


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