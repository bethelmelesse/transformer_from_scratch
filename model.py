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
print(tokenized)

token_input_ids = torch.LongTensor(tokenized["input_ids"])
batch_size = len(contexts)
embedding_dim = 128
seq_length = len(token_input_ids[0])          # 16 after padding for this examples 


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # task 1: Intitialize w for query, key and value
        self.w_query = torch.rand(embedding_dim, embedding_dim)        # shape of w = (128 * 256)
        self.w_key = torch.rand(embedding_dim, embedding_dim) 
        self.w_value = torch.rand(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x):                        # x = 2, 16, 128
        # task 2: get query, key, value
        query = torch.matmul(x, self.w_query)    # shape of q, k, v = 2 * 16 * 256
        key = torch.matmul(x, self.w_key)          
        value = torch.matmul(x, self.w_value)

        # task 3: multiply query by key
        product = torch.bmm(query, key.transpose(1,2))          # shape of product = 2 * 16 * 16

        # task 4: scale by root dk - dimensiion of key
        dk = key.size(dim=2)
        root_dk = math.sqrt(dk)
        scale = product / root_dk      # attention score 

        # task 5: softmax
        attention_probability = self.softmax(scale)      # shape of attention prob = 2 * 16 * 16

        # task 6: multiply attention prob by value and sum
        attention_output = torch.bmm(attention_probability, value)

        return attention_output

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        token_embeddings = self.embedding(x)     # x = (batch_size, seq_length) & (batch_size, seq_length, embedding_dim)  (2, 16, 128)
        attention_result = Attention()
        output = attention_result(token_embeddings)
        return output  

my_model = Model()
my_model(token_input_ids)

print()