from transformers import BertTokenizer, AutoTokenizer
import torch
import torch.nn as nn
import math

# common
BATCH_SIZE = 2
EMBEDDING_DIM = 128

''' ------------------------------------------------------Attention----------------------------------------------------------------------------------'''

class Attention_layer(nn.Module):
    def __init__(self):
        super().__init__()
        # task 1: Intitialize w for query, key and value
        self.w_query = torch.rand(EMBEDDING_DIM, EMBEDDING_DIM)        # shape of w = (128 * 128)
        self.w_key = torch.rand(EMBEDDING_DIM, EMBEDDING_DIM) 
        self.w_value = torch.rand(EMBEDDING_DIM, EMBEDDING_DIM)

        self.num_heads = 8
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, input_embeddings, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding, masked):      # x = 2, 16, 128
        seq_length_source = len(token_attention_masks_source[0])
        seq_length_target = len(token_attention_masks_target[0])
        
        if (masked == False):
            seq_length = seq_length_source 
        else:
            seq_length = seq_length_target 

        num_heads = self.num_heads
        
        # task 2: get query, key, value      
        key = torch.matmul(input_embeddings, self.w_key)          
        value = torch.matmul(input_embeddings, self.w_value)

         # task 3: split query, key and value into attention heads and reshape 
        head_dim = int(EMBEDDING_DIM / self.num_heads)          # 16

         # task 3.1: for key
        new_key = torch.reshape(key, (BATCH_SIZE, seq_length, head_dim, num_heads))       # encoder = 2, 16, 16, 8   decoder = 2, 19, 16, 8    cross = dencoder
        new_key = torch.transpose(new_key, 2, 3)                                               # encoder = 2, 16, 8, 16   decoder = 2, 19, 8, 16
        new_key = torch.transpose(new_key, 1, 2)                                               # encoder = 2, 8, 16, 16   decoder = 2, 8, 19, 16
        new_key = torch.reshape(new_key, (BATCH_SIZE * num_heads, seq_length,head_dim))        # encoder = 16, 16, 16     decoder = 16, 19, 16

         # task 3.2: for value
        new_value = torch.reshape(value, (BATCH_SIZE, seq_length, head_dim, num_heads))       # encoder = 2, 16, 16, 8   decoder = 2, 19, 16, 8  cross = dencoder
        new_value = torch.transpose(new_value, 2, 3)                                          # encoder = 2, 16, 8, 16   decoder = 2, 19, 8, 16
        new_value = torch.transpose(new_value, 1, 2)                                          # encoder = 2, 8, 16, 16   decoder = 2, 8, 19, 16
        new_value = torch.reshape(new_value, (BATCH_SIZE * num_heads, seq_length, head_dim))   # encoder = 16, 16, 16     decoder = 16, 19, 16

        # task 3.3: for query

        if (encoder_output_embedding == None):
            query = torch.matmul(input_embeddings, self.w_query)    # shape of q, k, v = 2 * 16 * 128
        else:
            query = torch.matmul(encoder_output_embedding, self.w_query) 
            seq_length = seq_length_source                  # 16   cross

        # Reshape to Batch, seq_length, head_dim, num_head
        new_query = torch.reshape(query, (BATCH_SIZE, seq_length, head_dim, num_heads))      # encoder = 2, 16, 16, 8   decoder = 2, 19, 16, 8    cross = encoder
        # swap -->  Batch, seq_length, num_head, head_dim
        new_query = torch.transpose(new_query, 2, 3)                                         # encoder = 2, 16, 8, 16   decoder = 2, 19, 8, 16
        # swap -->  Batch, num_head, seq_length, head_dim
        new_query = torch.transpose(new_query, 1, 2)                                         # encoder = 2, 8, 16, 16   decoder = 2, 8, 19, 16
        # reshape again --->
        new_query = torch.reshape(new_query, (BATCH_SIZE * num_heads, seq_length, head_dim))  # encoder = 16, 16, 16    decoder = 16, 19, 16

       
        # task 4: multiply query by key
        product_across_heads = torch.bmm(new_query, new_key.transpose(1, 2))             # encoder = 16, 16, 16        decoder = 16, 19, 19    cross = 16, 16, 19

        # task 5: scale the product 
        d_head_dim = new_key.size(dim=2)                                  # encoder = 16        decoder = 16       cross = 16
        root_d_head_dim = math.sqrt(d_head_dim)                            # encoder = 4        decoder = 4        cross = 4
        scale = product_across_heads / root_d_head_dim      # attention score    # encoder = 16, 16, 16       decoder = 16, 19, 19   # cross = 16, 16, 19

        # task 6: include attention mask  for the encoder
        if (masked == False):
            encoder_attention_mask = token_attention_masks_source.clone()
            encoder_attention_mask[encoder_attention_mask == 0] = -1000                                            # shape = 2 * 16
            encoder_attention_mask[encoder_attention_mask == 1] = 0                                                # shape = 2 * 16
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)                                           # shape = 2 * 1 * 16
            encoder_attention_mask = encoder_attention_mask.repeat_interleave(repeats=num_heads, dim=1)           # shape = 2 * 8 * 16
            encoder_attention_mask = encoder_attention_mask.reshape(BATCH_SIZE * num_heads, seq_length)           # shape = 16 * 16

            attention_score = scale + encoder_attention_mask                          # encoder = 16, 16, 16

        else:
            # decoder_attention_mask = token_attention_masks_source.clone()
            attention_score = scale                                                  # decoder = 16, 19, 19        # cross = 16, 16, 19

       
        # task 7: softmax the attention score 
        attention_probabilty_across_heads = self.softmax(attention_score)            # encoder = 16, 16, 16       decoder = 16, 19, 19    cross = 16, 16, 19
        
        # task 8: multiply attention prob by value and sum
        prob_by_value_and_sum = torch.bmm(attention_probabilty_across_heads, new_value)   # encoder = 16, 16, 16    decoder = 16, 19, 19    cross = 16, 16, 16
        prob_by_value_and_sum = torch.reshape(prob_by_value_and_sum, (BATCH_SIZE, num_heads, seq_length, head_dim))     # encoder = 2, 8, 16, 16   decoder = 2, 8, 19, 16           cross = 2, 8, 16, 16

        # task 9: reorder 
        attention_output = torch.transpose(prob_by_value_and_sum , 1, 2)        # encoder = 2, 16, 8, 16    decoder = 2, 19, 8, 16   cross = 2, 16, 8, 16 
        attention_output = torch.transpose(attention_output, 2, 3)              # encoder = 2, 16, 16, 8    decoder = 2, 19, 16, 8   cross = 2, 16, 16, 8 
        attention_output = torch.reshape(attention_output, (BATCH_SIZE, seq_length, head_dim * num_heads))    # encoder = 2, 16, 128   decoder =  2, 19, 128  cross = 2, 16, 128  

        return attention_output                       

''' ------------------------------------------------------Embeddings----------------------------------------------------------------------------------'''

class Positional_embed(nn.Module):
    def __init__(self, max_seq_length):
        super().__init__()
        self.posit_embedding = nn.Embedding(max_seq_length, EMBEDDING_DIM)
        
    def forward(self, seq_length):
        posit_embed_init = torch.arange(0, seq_length)
        positional_embeddings = self.posit_embedding(posit_embed_init).unsqueeze(0)
        return positional_embeddings       # shape = 1 * 16 * 128

class Input_embedding(nn.Module):
    def __init__(self, vocab_size, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.posit_embed = Positional_embed(max_seq_length)

    def forward(self, token_input_ids):
        token_embeddings = self.embedding(token_input_ids)     # x = (batch_size, seq_length_source) & (batch_size, seq_length_source, embedding_dim)  (2, 16, 128)
        seq_length = len(token_input_ids[0])
        position_embeddings = self.posit_embed(seq_length)
        token_embeddings_with_posit = token_embeddings + position_embeddings

        return token_embeddings_with_posit

''' ----------------------------------------------------Transformer layers-----------------------------------------------------------------------------'''

class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=2)
        mean = torch.unsqueeze(mean, 2)
        variance= torch.var(x, dim=2)
        variance = torch.unsqueeze(variance, 2)
        layer_norm = (x - mean) / variance
        return layer_norm

class Encoder_transformer_layer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.attention = Attention_layer()                                           # step 1 - self-attention
        self.linear_1 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)                     # step 2 - linear
        self.layer_norm_1 = LayerNormalization()                                    # step 3 - layer norm
        self.linear_2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * 4)                 # step 4 - linear
        self.relu = nn.ReLU()                                                       # step 5 - relu
        self.linear_3 = nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM)                 # step 6 - linear
        self.layer_norm_2 = LayerNormalization()                                    # step 7 - layer norm

    def forward(self, input_embeddings_source, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding=None):
        
        attention_output = self.attention(input_embeddings_source, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding, masked=False)
    
        linear_layer_1 = self.linear_1(attention_output)
        residual_output_1 = linear_layer_1 + input_embeddings_source         # shape = 2 * 16 * 128

        layer_norm_output_1 = self.layer_norm_1(residual_output_1)      # shape = 2 * 16 * 128

        linear_layer_2 = self.linear_2(layer_norm_output_1)           # shape = 2 * 16 * 512
        relu_output = self.relu(linear_layer_2)                    # shape = 2 * 16 * 512

        linear_layer_3 = self.linear_3(relu_output)                  # shape = 2 * 16 * 128
        residual_output_2 = linear_layer_3 + layer_norm_output_1        # shape = 2 * 16 * 128
        layer_norm_output_2 = self.layer_norm_2(residual_output_2)      # shape = 2 * 16 * 128

        return layer_norm_output_2 

class Decoder_transformer_layer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.masked_self_attention = Attention_layer()                               # step 1 - self-attention - masked self attention ---- make it masked
        self.linear_1 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)                     # step 2 - linear

        self.cross_attention = Attention_layer()                                     # step 3 - self-attention - cross attention --- make it masked
        self.linear_2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * 4)                 # step 4 - linear

        self.layer_norm = LayerNormalization()                                      # step 5 - layer norm
        self.linear_3 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * 4)                 # step 6 - linear
        self.relu = nn.ReLU()                                                       # step 7 - relu
        self.linear_4 = nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM)                 # step 8 - linear

    def forward(self, input_embeddings_target, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding):
        
        masked_self_attention_output = self.masked_self_attention(input_embeddings_target, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding=None, masked=True)                            # decoder = 2 * 19 * 128
        linear_layer_1 = self.linear_1(masked_self_attention_output)           # decoder = 2 * 19 * 128
        residual_output_1 = linear_layer_1 + input_embeddings_target           # decoder = 2 * 19 * 128

        cross_attention_output = self.cross_attention(input_embeddings_target, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding=encoder_output_embedding, masked=True)   # decoder = 2 * 16 * 128
        linear_layer_2 = self.linear_2(cross_attention_output)            # decoder = 2 * 16 * 512

        residual_output_2 = linear_layer_2 + residual_output_1            # shape = 

        layer_norm_output = self.layer_norm(residual_output_2)            # shape = 
        linear_layer_3 = self.linear_3(layer_norm_output)                 # shape = 
        relu_output = self.relu(linear_layer_3)                           # shape = 
        linear_layer_4 = self.linear_4(relu_output)                       # shape = 
        residual_output_3 = linear_layer_4 + residual_output_2            # shape = 

        return residual_output_3

''' --------------------------------------------------------Main Model----------------------------------------------------------------------------------'''

class Model(nn.Module):
    def __init__(self, num_layers, vocab_size_source, max_seq_length_source, vocab_size_target, max_seq_length_target):
        super().__init__()
        self.input_embed_source = Input_embedding(vocab_size_source, max_seq_length_source)
        self.input_embed_target = Input_embedding(vocab_size_target, max_seq_length_target)
        self.num_layers = num_layers

        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(Encoder_transformer_layer(num_layers))

        self.decoder_layers = []
        for _ in range(num_layers):
            self.decoder_layers.append(Decoder_transformer_layer(num_layers))
  
         
    def forward(self, token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target):
        input_embeddings_source = self.input_embed_source(token_input_ids_source)
        input_embeddings_target = self.input_embed_target(token_input_ids_target)


        for n in range(self.num_layers):
            encoder_output = self.encoder_layers[n](input_embeddings_source, token_attention_masks_source, token_attention_masks_target)
            input_embeddings_source = encoder_output

        for n in range(self.num_layers): 
            decoder_output = self.decoder_layers[n](input_embeddings_target, token_attention_masks_source, token_attention_masks_target, encoder_output)
            input_embeddings_target = decoder_output

        return encoder_output

''' ----------------------------------------------------------Other-------------------------------------------------------------------------------------'''

''' -----------------------------------------------------------test-------------------------------------------------------------------------------------'''
def main():

    # Source Sequence
    source_context_1 ="My name is Bethel. I like to eat dirkosh."
    source_context_2 = "Obama is the US president. Then Trump is the old US president."
    source_contexts = [source_context_1, source_context_2]

    # Tokenizer - source
    tokenizer_source = BertTokenizer.from_pretrained('bert-base-cased')
    vocab_size_source = tokenizer_source.vocab_size
    tokenized_source = tokenizer_source(source_contexts, padding=True)

    # token_input_ids - source 
    token_input_ids_source = torch.LongTensor(tokenized_source["input_ids"])
    token_attention_masks_source = torch.LongTensor(tokenized_source["attention_mask"])

    max_seq_length_source  = 512

    # Target Sequence
    target_context_1 = "ስሜ ቤቴል ነው። ዲርኮሽ መብላት እወዳለሁ።"
    target_context_2 = "ኦባማ የአሜሪካ ፕሬዝዳንት ናቸው። ከዚያ ትራምፕ የድሮው የአሜሪካ ፕሬዝዳንት ናቸው።"
    target_contexts = [target_context_1, target_context_2]

    # Tokenizer - target
    tokenizer_target = AutoTokenizer.from_pretrained('xlm-roberta-base')
    vocab_size_target = tokenizer_target.vocab_size
    tokenized_target = tokenizer_target(target_contexts, padding=True)

    # token_input_ids - source 
    token_input_ids_target = torch.LongTensor(tokenized_target["input_ids"])
    token_attention_masks_target = torch.LongTensor(tokenized_target["attention_mask"])

    max_seq_length_target  = 512

    num_layers = 6
    my_model = Model(num_layers, vocab_size_source, max_seq_length_source, vocab_size_target, max_seq_length_target)
    a = my_model(token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target)

    print(a)
    print(a.shape)

    print()

if __name__ == "__main__":
    main()
