import argparse
import json
import os
from datasets import load_dataset
import torch
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader

from nlp.tokenizers import init_tokenizer



def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Json file with the tokenizer loading configuration")
    parser.add_argument("--tokenizer_dir",help="Directory that holds the tokenizer files")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    with open(args["config"],"r") as f:
        config = json.load(f)
    tokenizer_dir = args["tokenizer_dir"]
    output_dir = args["out"]

    return config, tokenizer_dir, output_dir


def preprocess_dataset(sample,tokenizer):
    encoded_input = tokenizer(
        sample["review_body"],
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return {
        "input_ids": encoded_input["input_ids"],
        "attention_mask": encoded_input["attention_mask"],
        "label": [label - 1 for label in sample["stars"]]
    }


def data_collate(batch,tokenizer):
    return tokenizer.pad(batch,return_tensors="pt",max_length=tokenizer.model_max_length)


def prepare_data_for_training(
        tokenizer: PreTrainedTokenizerFast,
        preprocess_dataset,
        data_collate,
        **data_config
    ):

    columns_to_return = ['input_ids', 'label', 'attention_mask']

    dataloaders = {}
    for split in ["train", "validation"]:
        dataset = load_dataset(**data_config[split]["loading_args"])
        dataset = dataset.map(
            lambda sample: preprocess_dataset(sample,tokenizer),
            **data_config[split]["mapping_args"]
        )
        dataset.set_format(type='torch', columns=columns_to_return)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=data_config[split]["batch_size"],
            shuffle=True,
            collate_fn=lambda batch: data_collate(batch,tokenizer)
        )

    return dataloaders


def main():
    config, tokenizer_dir, output_dir = parse_args()
    tokenizer = init_tokenizer(tokenizer_dir)

    # Train and Validation Data:
    dataloaders = prepare_data_for_training(
        tokenizer,
        preprocess_dataset,
        data_collate,
        **config
    )


if __name__ == "__main__":
    main()