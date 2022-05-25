from tokenizers.normalizers import Lowercase, Replace, Sequence as NormSeq
from tokenizers.pre_tokenizers import Whitespace, Sequence as PreTokSeq
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset

_special_tokens = {
    
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]"
}

config = {

    # Is a pretrained tokenizer:
    "is_pretrained": False,

    # Tokenizer model:
    "model": WordLevel(
        vocab=None,
        unk_token=_special_tokens["unk_token"]
    ),

    # Normalizer:
    "normalizer": NormSeq([Lowercase(), Replace("10","diez")]),

    # Pre-tokenizer:
    "pre-tokenizer": Whitespace(),

    # Tokenizer training args:
    "trainer": WordLevelTrainer(
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=list(_special_tokens.keys())
    ),

    "encoding_args": {
        "model_max_length": 512,
        "padding_side": "right",
        "truncation_side": "right",
        "model_input_names": ["input_ids", "attention_mask"],
        "bos_token": _special_tokens["bos_token"],
        "eos_token": _special_tokens["eos_token"],
        "sep_token": _special_tokens["sep_token"],
        "pad_token": _special_tokens["pad_token"],
        "cls_token": _special_tokens["cls_token"],
        "mask_token": _special_tokens["mask_token"],
        "additional_special_tokens": []
    }

}


def prepare_batch_iterator():
    # Load corpus
    corpus = load_dataset("amazon_reviews_multi","es",split="train")
    corpus_len = len(corpus)
    # Create batch_iterator
    batch_size = 32
    batch_iterator = (corpus[i:i+batch_size]["review_body"] for i in range(0,corpus_len,batch_size))
    return batch_iterator, corpus_len