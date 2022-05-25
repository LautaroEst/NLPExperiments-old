from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def train_tokenizer(config,corpus_or_batch_iterator,corpus_size):
    tokenizer = Tokenizer(config["model"])
    tokenizer.normalizer = config["normalizer"]
    tokenizer.pre_tokenizer = config["pre-tokenizer"]
    tokenizer.train_from_iterator(corpus_or_batch_iterator,trainer=config["trainer"],length=corpus_size)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,**config["encoding_args"])
    return tokenizer