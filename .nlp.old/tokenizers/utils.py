from transformers import AutoTokenizer
from datasets import load_dataset


def load_pretrained_tokenizer(**config):
    tokenizer = AutoTokenizer.from_pretrained(**config)
    return tokenizer


def prepare_corpus_to_train_tokenizer(**corpus_config):
    column = corpus_config.pop("use_column")
    batch_size = corpus_config.pop("train_batch_size")
    corpus = load_dataset(**corpus_config)
    corpus_len = len(corpus)
    
    batch_iterator = (corpus[i:i+batch_size][column] for i in range(0,len(corpus),batch_size))
    return batch_iterator, corpus_len


def load_tokenizer(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer