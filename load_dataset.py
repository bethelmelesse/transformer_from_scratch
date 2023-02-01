from config import *


def open_datasets(context_path):
    # dataset_url = 'https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/am-en.txt.zip'
    # import dload
    # dload.save_unzip(dataset_url)
    with open(context_path, encoding='utf8') as f:
        contexts = [source_context.strip() for source_context in f.readlines()][:EXAMPLE]
    return contexts


def tokenize_dataset(sets, tokenizer, max_seq_length):
    tokenized = tokenizer(sets, padding='max_length', max_length=max_seq_length, truncation=True)
    token_input_ids = torch.LongTensor(tokenized["input_ids"]).to(device=device)
    token_attention_masks = torch.LongTensor(tokenized["attention_mask"]).to(device=device)
    return token_input_ids, token_attention_masks


def train_and_test_split():
    source_contexts = open_datasets(SOURCE_CONTEXT_PATH)
    target_contexts = open_datasets(TARGET_CONTEXT_PATH)

    end = int(len(source_contexts) * TRAIN_RATIO)

    train_set_source = source_contexts[:end]
    test_set_source = source_contexts[end:]

    train_set_target = target_contexts[:end]
    test_set_target = target_contexts[end:]

    return train_set_source, train_set_target, test_set_source, test_set_target


def preprocess(set_source, set_target, tokenizer_source, tokenizer_target):
    token_input_ids_source, token_attention_masks_source = tokenize_dataset(set_source, tokenizer_source, MAX_SEQ_LENGTH_SOURCE)
    token_input_ids_target, token_attention_masks_target = tokenize_dataset(set_target, tokenizer_target, MAX_SEQ_LENGTH_TARGET)
    return token_input_ids_source, token_attention_masks_source, token_input_ids_target, token_attention_masks_target
