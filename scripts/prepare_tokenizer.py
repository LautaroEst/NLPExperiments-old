import argparse
import json
from nlp.tokenizers import load_pretrained_tokenizer, train_tokenizer
from nlp.utils import import_configs_objs


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Python file with the tokenizer loading configuration")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Extract config and output directory
    output_dir = args["out"]
    config_file = args["config"]

    return config_file, output_dir


def main():
    config_file, output_dir = parse_args()
    config_objs = import_configs_objs(config_file)
    config_dict = config_objs["config"]

    is_pretrained = config_dict.pop("is_pretrained")
    if is_pretrained:
        tokenizer = load_pretrained_tokenizer(**config_dict)
    else:
        prepare_batch_iterator = config_objs["prepare_batch_iterator"]
        batch_iterator, corpus_len = prepare_batch_iterator()
        tokenizer = train_tokenizer(config_dict,batch_iterator,corpus_len)

    tokenizer.save_pretrained(f"{output_dir}")


if __name__ == "__main__":
    main()