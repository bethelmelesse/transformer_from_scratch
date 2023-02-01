import time

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import *
from enc_dec_model import Model
from load_dataset import preprocess, train_and_test_split


def main():
    print()
    tic = time.time()
    train_set_source, train_set_target, test_set_source, test_set_target = train_and_test_split()
    tokenizer_source = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE_NAME)
    tokenizer_target = AutoTokenizer.from_pretrained(TOKENIZER_TARGET_NAME)

    vocab_size_source = tokenizer_source.vocab_size
    vocab_size_target = tokenizer_target.vocab_size
    vocab_target = tokenizer_target.vocab

    my_model = Model(vocab_size_source, vocab_size_target, vocab_target).to(device=device)

    optimizer = torch.optim.Adam(my_model.parameters(), lr = LR, weight_decay=WEIGHT_DECAY)     # select the optimizer

    def train():
        token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target = preprocess(train_set_source, train_set_target,tokenizer_source, tokenizer_target)

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
        token_input_ids_source, token_attention_masks_source,token_input_ids_target, token_attention_masks_target = preprocess(test_set_source, test_set_target, tokenizer_source, tokenizer_target)

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
                    translated.append(tokenizer_target.decode(predicted[i]))
        
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
 