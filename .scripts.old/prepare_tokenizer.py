import argparse
import json
from nlp.tokenizers import load_pretrained_tokenizer, train_tokenizer_from_iterable, prepare_corpus_to_train_tokenizer



def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Json file with the tokenizer loading configuration")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Extract config and output directory
    output_dir = args["out"]
    is_pretrained = "pretrained" in args["config"]
    with open(args["config"],"r") as f:
        config = json.load(f)

    return output_dir, config, is_pretrained


def main():
    output_dir, config, is_pretrained = parse_args()

    if is_pretrained:
        tokenizer = load_pretrained_tokenizer(**config)
    else:
        corpus_config = config.pop("train_corpus_args")
        batch_iterator, corpus_len = prepare_corpus_to_train_tokenizer(**corpus_config)
        tokenizer = train_tokenizer_from_iterable(batch_iterator,corpus_len,**config)

    tokenizer.save_pretrained(f"{output_dir}")


if __name__ == "__main__":
    main()