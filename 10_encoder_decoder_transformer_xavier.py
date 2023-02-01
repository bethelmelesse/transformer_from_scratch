from transformers import BertTokenizer, AutoTokenizer
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import numpy as np
import dload
import evaluate
import time 

tic = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# common
BATCH_SIZE = 2
EMBEDDING_DIM = 128
SOURCE_CONTEXT_PATH = './am-en.txt/CCAligned.am-en.en'
TARGET_CONTEXT_PATH = './am-en.txt/CCAligned.am-en.am'
TOKENIZER_SOURCE = BertTokenizer.from_pretrained('bert-base-cased')
TOKENIZER_TARGET = AutoTokenizer.from_pretrained('xlm-roberta-base')
TRAIN_RATIO = 0.8
MAX_SEQ_LENGTH_SOURCE  = 512
MAX_SEQ_LENGTH_TARGET  = 512
NUM_LAYERS = 2
LR = 0.00001
WEIGHT_DECAY=1e-5
EPOCHES = 2
# dataset_url = 'https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/am-en.txt.zip'
# dload.save_unzip(dataset_url) 
EXAMPLE = 10
print()

''' --------------------------------------------------------DATASET-------------------------------------------------------------------------------------'''
def open_datasets(context_path):
    with open(context_path, encoding='utf8') as f:
        contexts = [source_context.strip() for  source_context in f.readlines()][:EXAMPLE]
    return contexts

def tokenize_dataset(sets, tokenizer):
    tokenized = tokenizer(sets, padding='max_length')
    token_input_ids = torch.LongTensor(tokenized["input_ids"]).to(device=device)
    token_attention_masks = torch.LongTensor(tokenized["attention_mask"]).to(device=device)
    return token_input_ids, token_attention_masks

def train_and_test_split():
    source_contexts = open_datasets(SOURCE_CONTEXT_PATH)
    target_contexts = open_datasets(TARGET_CONTEXT_PATH)

    end = int(len(source_contexts) * TRAIN_RATIO)
    
    train_set_source = source_contexts[:end]
    test_set_source =  source_contexts[end:]

    train_set_target = target_contexts[:end]
    test_set_target = target_contexts[end:]

    return train_set_source, train_set_target, test_set_source, test_set_target

def preprocess(set_source, set_target):
    token_input_ids_source, token_attention_masks_source = tokenize_dataset(set_source, TOKENIZER_SOURCE)
    token_input_ids_target, token_attention_masks_target = tokenize_dataset(set_target, TOKENIZER_TARGET)
    return token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target

''' ------------------------------------------------------Embeddings----------------------------------------------------------------------------------'''

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
        token_embeddings = self.embedding(token_input_ids)     # x = (batch_size, seq_length_source) & (batch_size, seq_length_source, embedding_dim)  (2, 16, 128)
        seq_length = len(token_input_ids[0])
        position_embeddings = self.posit_embed(seq_length)
        token_embeddings_with_posit = token_embeddings + position_embeddings

        return token_embeddings_with_posit

''' ------------------------------------------------------Attention----------------------------------------------------------------------------------'''

class Attention_layer(nn.Module):
    def __init__(self):
        super().__init__()
        # task 1: Intitialize w for query, key and value
        self.w_query = nn.Parameter(torch.rand(EMBEDDING_DIM, EMBEDDING_DIM))       # shape of w = (128 * 128)
        self.w_query = torch.nn.init.xavier_normal_(self.w_query)
        self.w_key =  nn.Parameter(torch.rand(EMBEDDING_DIM, EMBEDDING_DIM)) 
        self.w_key = torch.nn.init.xavier_normal_(self.w_key)  
        self.w_value =  nn.Parameter(torch.rand(EMBEDDING_DIM, EMBEDDING_DIM)) 
        self.w_value = torch.nn.init.xavier_normal_(self.w_value)  

        self.num_heads = 8
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, input_embeddings, token_attention_masks_source, token_attention_masks_target, encoder_output_embedding, attention_type):      # x = 2, 16, 128  

        seq_length_source = len(token_attention_masks_source[0])
        seq_length_target = len(token_attention_masks_target[0])

        num_heads = self.num_heads
        head_dim = int(EMBEDDING_DIM / self.num_heads)          # 16

        def query(embeddings, seq_length):  
            query = torch.matmul(embeddings, self.w_query)    # shape of q, k, v = 2 * 16 * 128         cross = 2, 18, 128
            new_query = torch.reshape(query, (BATCH_SIZE, seq_length, head_dim, num_heads))      # encoder = 2, 16, 16, 8   decoder = 2, 18, 16, 8    cross = decoder
            new_query = torch.transpose(new_query, 2, 3)                                         # encoder = 2, 16, 8, 16   decoder = 2, 18, 8, 16
            new_query = torch.transpose(new_query, 1, 2)                                         # encoder = 2, 8, 16, 16   decoder = 2, 8, 18, 16
            new_query = torch.reshape(new_query, (BATCH_SIZE * num_heads, seq_length, head_dim))  # encoder = 16, 16, 16    decoder = 16, 18, 16
            return new_query

        def key(embeddings, seq_length):
            key = torch.matmul(embeddings, self.w_key)
            new_key = torch.reshape(key, (BATCH_SIZE, seq_length, head_dim, num_heads))       # encoder = 2, 16, 16, 8   decoder = 2, 18, 16, 8    cross = encoder
            new_key = torch.transpose(new_key, 2, 3)                                               # encoder = 2, 16, 8, 16   decoder = 2, 18, 8, 16
            new_key = torch.transpose(new_key, 1, 2)                                               # encoder = 2, 8, 16, 16   decoder = 2, 8, 18, 16
            new_key = torch.reshape(new_key, (BATCH_SIZE * num_heads, seq_length,head_dim))        # encoder = 16, 16, 16     decoder = 16, 18, 16
            return new_key


        def value(embeddings, seq_length):
            value = torch.matmul(embeddings, self.w_value)
            new_value = torch.reshape(value, (BATCH_SIZE, seq_length, head_dim, num_heads))       # encoder = 2, 16, 16, 8   decoder = 2, 18, 16, 8  cross = encoder
            new_value = torch.transpose(new_value, 2, 3)                                          # encoder = 2, 16, 8, 16   decoder = 2, 18, 8, 16
            new_value = torch.transpose(new_value, 1, 2)                                          # encoder = 2, 8, 16, 16   decoder = 2, 8, 18, 16
            new_value = torch.reshape(new_value, (BATCH_SIZE * num_heads, seq_length, head_dim))   # encoder = 16, 16, 16     decoder = 16, 18, 16
            return new_value 

        def scaled_product(query, key):
            product_across_heads = torch.bmm(query, key.transpose(1, 2))             # encoder = 16, 16, 16        decoder = 16, 18, 18    cross = 16, 18, 16
            d_head_dim = key.size(dim=2)                                  # encoder = 16        decoder = 16       cross = 16
            root_d_head_dim = math.sqrt(d_head_dim)                            # encoder = 4        decoder = 4        cross = 4
            scale = product_across_heads / root_d_head_dim      # attention score    # encoder = 16, 16, 16       decoder = 16, 18, 18   # cross = 16, 18, 16
            return scale 

        def attention_mask(attention_mask, seq_length):
            attention_mask[attention_mask == 0] = -1000                                            # encoder = 2 * 16            # cross = 2 * 18
            attention_mask[attention_mask == 1] = 0                                                # encoder = 2 * 16            # cross = 2 * 18
            attention_mask = attention_mask.unsqueeze(1)                                           # encoder = 2 * 1 * 16        # cross = 2 * 1 * 18
            attention_mask = attention_mask.repeat_interleave(repeats=num_heads, dim=1)            # encoder = 2 * 8 * 16        # cross = 2 * 8 * 18
            attention_mask = attention_mask.reshape(BATCH_SIZE * num_heads, seq_length)            # encoder = 16 * 16           # cross = 16 * 18
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
            attention_probabilty_across_heads = self.softmax(attention_score)            # encoder = 16, 16, 16       decoder = 16, 18, 18    cross = 16, 18, 16
            prob_by_value_and_sum = torch.bmm(attention_probabilty_across_heads, value)   # encoder = 16, 16, 16    decoder = 16, 18, 18    cross = 16, 18, 16
            prob_by_value_and_sum = torch.reshape(prob_by_value_and_sum, (BATCH_SIZE, num_heads, seq_length, head_dim))
            attention_output = torch.transpose(prob_by_value_and_sum , 1, 2)        # encoder = 2, 16, 8, 16    decoder = 2, 18, 8, 16   cross = 2, 18, 8, 16
            attention_output = torch.transpose(attention_output, 2, 3)              # encoder = 2, 16, 16, 8    decoder = 2, 18, 16, 8   cross = 2, 18, 16, 8 
            attention_output = torch.reshape(attention_output, (BATCH_SIZE, seq_length, head_dim * num_heads))    # encoder = 2, 16, 128   decoder =  2, 18, 128  cross = 2, 18, 128 
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

            attention_score = scaled_product  + decoder_attention_mask()                                   # decoder = 16, 18, 18 
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

''' --------------------------------------------------------Main Model----------------------------------------------------------------------------------'''

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
  
         
    def forward(self, token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target, is_training=True):
        input_embeddings_source = self.input_embed_source(token_input_ids_source)
        input_embeddings_target = self.input_embed_target(token_input_ids_target[:,:-1])
        token_attention_masks_target_without_end = token_attention_masks_target[:,:-1]


        for n in range(NUM_LAYERS):
            encoder_output = self.encoder_layers[n](input_embeddings_source, token_attention_masks_source, token_attention_masks_target_without_end)
            input_embeddings_source = encoder_output

        for n in range(NUM_LAYERS): 
            decoder_output = self.decoder_layers[n](input_embeddings_target, token_attention_masks_source, token_attention_masks_target_without_end, encoder_output)
            input_embeddings_target = decoder_output

        # print(decoder_output.shape)          # ([2, 18, 128])

                                                               
        look_up_table = self.lookup_table.weight.transpose(0, 1)          # ([250002, 128]) ---> ([128, 250002])
        dot_product = torch.matmul(decoder_output, look_up_table)         # ([2, 18, 250002])

        ground_truth = token_input_ids_target[:,1:].reshape(-1)
        # This will be used during inference
        if is_training==True:
            index_highest_prob = None
            loss = self.loss(dot_product.view(-1, dot_product.shape[2]), ground_truth)
            seq_len_target = token_input_ids_target[:,:-1].shape[1]
            loss = loss.view(-1, seq_len_target)
            loss = loss * token_attention_masks_target_without_end
            loss = torch.sum(loss)/torch.sum(token_attention_masks_target_without_end)
            
        else:
            loss = None
            softmax = self.softmax(dot_product)                                        # ([2, 18, 250002])
            index_highest_prob = torch.argmax(softmax, dim=2)
            
        return index_highest_prob, loss

''' -----------------------------------------------------------main-------------------------------------------------------------------------------------'''
def main():
    train_set_source, train_set_target, test_set_source, test_set_target = train_and_test_split()

    vocab_size_source = TOKENIZER_SOURCE.vocab_size
    vocab_size_target = TOKENIZER_TARGET.vocab_size
    vocab_target = TOKENIZER_TARGET.vocab

    my_model = Model(vocab_size_source, vocab_size_target, vocab_target).to(device=device)

    optimizer = torch.optim.Adam(my_model.parameters(), lr = LR, weight_decay=WEIGHT_DECAY)     # select the optimizer

    def train():
        token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target = preprocess(train_set_source, train_set_target)

        steps = int(len(token_input_ids_source)/BATCH_SIZE)
        for epoch in tqdm(range(EPOCHES)):
            start = 0
            end = BATCH_SIZE
            for step in tqdm(range(steps)): 
                predicted, loss = my_model(token_input_ids_source[start:end,], token_attention_masks_source[start:end,], token_input_ids_target[start:end,], token_attention_masks_target[start:end,], is_training=True)
                start = end
                end = start + BATCH_SIZE

                loss.backward()
                print(loss.item())
                optimizer.step()
                optimizer.zero_grad           
            
    def test():
        token_input_ids_source, token_attention_masks_source,token_input_ids_target, token_attention_masks_target = preprocess(test_set_source, test_set_target)

        my_model.eval()
        translated = []
        with torch.no_grad():
            steps = int(len(token_input_ids_target)/BATCH_SIZE)
            start = 0
            end = BATCH_SIZE
            for step in tqdm(range(steps)): 
                predicted, loss = my_model(token_input_ids_source[start:end,], token_attention_masks_source[start:end,], token_input_ids_target[start:end,], token_attention_masks_target[start:end,], is_training=False)
                start = end
                end = start + BATCH_SIZE
                
                # print(predicted)
                predicted = predicted.tolist()
                # print(predicted)
                
                for i in range(len(predicted)):
                    # translated_tokens = TOKENIZER_TARGET.convert_ids_to_tokens(predicted[i])
                    # translated.append(TOKENIZER_TARGET.convert_tokens_to_string(translated_tokens))
                    translated.append(TOKENIZER_TARGET.decode(predicted[i]))
        
        return translated

    def evaluation(preds, target):
        bleu = evaluate.load("bleu")
        evaluation_result = bleu.compute(predictions=preds, references=target)
        return evaluation_result

    
    def extractDigits(lst):
        return list(map(lambda el:[el], lst))

    def to_print(preds, set_source, set_target):
        print('\33[34m' + f"\nTRANSLATED: {preds}\n" + '\033[0m')
        print('\033[91m' + f"SOURCE: {set_source}\n" + '\033[0m')
        target = extractDigits(set_target)
        print('\33[32m' + f"TARGET: {target}\n" + '\033[0m')
        print('\33[36m' + f"Evaluation: {evaluation(preds, target)}\n" + '\033[0m')
    
    train()
    preds_test = test()
    to_print(preds_test, test_set_source, test_set_target)

    toc = time.time()
    print(f"time took: {toc-tic} sec\n")
    
if __name__ == "__main__":
    main()
 