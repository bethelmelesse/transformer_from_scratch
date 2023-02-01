import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2
EMBEDDING_DIM = 128
SOURCE_CONTEXT_PATH = './am-en.txt/CCAligned.am-en.en'
TARGET_CONTEXT_PATH = './am-en.txt/CCAligned.am-en.am'
TOKENIZER_SOURCE_NAME = 'bert-base-cased'
TOKENIZER_TARGET_NAME = 'xlm-roberta-base'
TRAIN_RATIO = 0.8
MAX_SEQ_LENGTH_SOURCE  = 512
MAX_SEQ_LENGTH_TARGET  = 512
NUM_LAYERS = 2
LR = 0.00001
WEIGHT_DECAY=1e-5
EPOCHES = 2

EXAMPLE = 10