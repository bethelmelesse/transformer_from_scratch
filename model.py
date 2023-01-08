from transformers import BertTokenizer
import torch
import torch.nn as nn

print()

context_1 ="My name is Bethel. I name to eat dirkosh."
context_2 = "Obama is the US president. Then Trump is the old US president."

contexts = [context_1, context_2]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
vocab_size = tokenizer.vocab_size

tokenized = tokenizer(contexts, padding=True)
print(tokenized)
x = torch.LongTensor(tokenized["input_ids"])
batch_size = len(contexts)
embedding_dim = 128

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        output = self.embedding(x)         # torch.Size([batch_size, sequence_length]) 
        return output                      # (batch_size, seq_length, embedding_dim)

my_model = Model()
my_model(x)

print()