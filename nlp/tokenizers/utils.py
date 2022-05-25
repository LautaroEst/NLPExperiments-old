from transformers import AutoTokenizer


def load_pretrained_tokenizer(**config):
    tokenizer = AutoTokenizer.from_pretrained(**config)
    return tokenizer


def load_tokenizer(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer