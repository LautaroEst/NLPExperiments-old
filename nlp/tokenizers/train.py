import tokenizers
from tokenizers import Tokenizer, models, trainers
from transformers import PreTrainedTokenizerFast

def _parse_list_of_objects(lst,list_type):
    if list_type == "normalizer":
        module = tokenizers.normalizers
    elif list_type == "pre_tokenizer":
        module = tokenizers.pre_tokenizers
    
    objects_list = [getattr(module,obj["name"])(**obj["params"]) for obj in lst]

    return objects_list



def _init_wordlevel_tokenizer_and_trainer(**kwargs):

    # Tokenizer model initialization:
    model_args = kwargs.pop("model_args").copy()
    vocab = model_args.pop("vocab",None)
    unk_token = model_args.pop("unk_token","[UNK]")

    tokenizer = Tokenizer(
        models.WordLevel(
            vocab=vocab,
            unk_token=unk_token
        )
    )

    # Training initialization:
    trainer_args = kwargs.pop("trainer_args").copy()
    vocab_size = trainer_args.pop("vocab_size",30000)
    min_frequency = trainer_args.pop("min_frequency",0)
    show_progress = trainer_args.pop("show_progress",True)
    special_tokens = trainer_args.pop("special_tokens",[])
    limit_alphabet = trainer_args.pop("limit_alphabet",None)
    initial_alphabet = trainer_args.pop("initial_alphabet",[])
    continuing_subword_prefix = trainer_args.pop("continuing_subword_prefix",'##')
    end_of_word_suffix = trainer_args.pop("end_of_word_suffix",None)

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=special_tokens,
        limit_alphabet=limit_alphabet,
        initial_alphabet=initial_alphabet,
        continuing_subword_prefix=continuing_subword_prefix,
        end_of_word_suffix=end_of_word_suffix
    )

    return tokenizer, trainer


## TO DO: Train WordPiece, BPE, Unigram
## TO DO: Decoder (an visualizer?) for all models


def train_tokenizer_from_iterable(
    corpus_or_batch_iterator,
    corpus_size,
    model="WordLevel",
    normalization=None,
    pre_tokenization=None,
    **kwargs
):

    # Tokenizer model:
    if model == "WordLevel":
        tokenizer, trainer = _init_wordlevel_tokenizer_and_trainer(**kwargs)
    else:
        ValueError(f"Tokenizer model {model} not supported.")
    
    # Normalizer:
    if isinstance(normalization,list):
        tokenizer.normalizer = tokenizers.normalizers.Sequence(_parse_list_of_objects(normalization,list_type="normalizer"))
    elif normalization is None:
        pass
    else:
        raise ValueError("Invalid normalizer.")
    
    # Pre-tokenizer:
    if isinstance(pre_tokenization,list):
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(_parse_list_of_objects(pre_tokenization,list_type="pre_tokenizer"))
    elif pre_tokenization is None:
        pass
    else:
        raise ValueError("Invalid pre-tokenizer.")

    # Training:
    tokenizer.train_from_iterator(corpus_or_batch_iterator,trainer=trainer,length=corpus_size)
    
    # Cast to transformers's tokenizer
    kwargs["encoding_args"]["unk_token"] = kwargs["model_args"]["unk_token"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,**kwargs["encoding_args"])

    return tokenizer


