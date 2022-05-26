import argparse

from nlp.tokenizers import load_tokenizer
from nlp.features import load_features_extractor
from datasets import load_from_disk
from nlp.utils import import_configs_objs
from nlp.training import train_sequence_classifier
from nlp.models import 

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config",help="Config file with training hyperparams.")
    parser.add_argument("--model_config",help="Config file with model hyperparams.")
    parser.add_argument("--tokenizer_files_dir",help="The Tokenizer directory that holds all data to load the Tokenizer.")
    parser.add_argument("--features_files_dir",help="The features directory that holds all data to load the Features Extractor.")
    parser.add_argument("--data_files_dir",help="The data directory that holds all mapped and tokenized data.")
    parser.add_argument("--out",help="The directory to leave the training files (checkpoint, logs, etc.)")
    args = vars(parser.parse_args())

    training_config_objs = import_configs_objs(args.pop("training_config"))
    model_config_objs = import_configs_objs(args.pop("model_config"))
    output_dir = args.pop("out")
    directories = {key.split("_files_dir")[0]: dir_path for key, dir_path in args.items()}

    return training_config_objs, model_config_objs, output_dir, directories


def load_data(data_path,tokenizer,**training_config):

    data_collate = training_config.pop("data_collate")
    dataset = load_from_disk(data_path)
    dataloaders = {}
    for split in ["train","validation"]:
        dataset[split].set_format(type='torch', columns=list(dataset[split].features.keys()))
        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=training_config["config"][f"{split}_batch_size"],
            shuffle=True,
            collate_fn=lambda batch: data_collate(batch,tokenizer)
        )

    return dataloaders


def main():
    training_config_objs, model_config_objs, output_dir, directories = parse_args()

    tokenizer = load_tokenizer(directories["tokenizer"])
    features_extractor = load_features_extractor(tokenizer,directories["features"])
    dataloaders = load_data(directories["data"],tokenizer,**training_config_objs)


    results_history = train_classification_model(
        features_extractor=features_extractor,
        main_model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["validation"],
        **args["training_config"]
    )



if __name__ == "__main__":
    main()