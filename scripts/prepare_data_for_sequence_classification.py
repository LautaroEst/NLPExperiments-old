import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from nlp.tokenizers import load_tokenizer
from transformers import PreTrainedTokenizerFast
from nlp.utils import import_configs_objs
from datasets import load_dataset, DatasetDict


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Json file with the tokenizer loading configuration")
    parser.add_argument("--tokenizer_dir",help="Directory that holds the tokenizer files")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    config_objs = import_configs_objs(args["config"])
    tokenizer_dir = args["tokenizer_dir"]
    output_dir = args["out"]

    return config_objs, tokenizer_dir, output_dir


def prepare_data_for_training(
        tokenizer: PreTrainedTokenizerFast,
        preprocess_dataset,
        output_dir,
        **data_config
    ):

    data_dict = {}
    for split in ["train", "validation"]:
        dataset = load_dataset(**data_config[split]["loading_args"])
        columns_to_remove = list(
            set(dataset.features.keys()) - \
            set(tokenizer.model_input_names) - \
            {"label"}
        )
        dataset = dataset.map(
            lambda sample: preprocess_dataset(sample,tokenizer),
            **data_config[split]["mapping_args"]
        )
        dataset = dataset.remove_columns(columns_to_remove)
        data_dict[split] = dataset

    DatasetDict(data_dict).save_to_disk(output_dir)



def main():
    config_objs, tokenizer_dir, output_dir = parse_args()
    tokenizer = load_tokenizer(tokenizer_dir)

    # Train and Validation Data:
    dataloaders = prepare_data_for_training(
        tokenizer,
        config_objs["preprocess_dataset"],
        output_dir,
        **config_objs["config"]
    )

    
if __name__ == "__main__":
    main()