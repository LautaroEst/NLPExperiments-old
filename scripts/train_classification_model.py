import argparse
import json

from nlp.tokenizers import init_tokenizer
from nlp.features import init_features_extractor
from nlp.models import init_model
from nlp.data import prepare_data_for_training
from nlp.trainers import train_classification_model


def parse_args():
    parser = argparse.ArgumentParser()

    # Tokenizer directory:
    parser.add_argument("--tokenizer_dir",help="Directory of the trained tokenizer")

    # Output directory:
    parser.add_argument("--out",help="Output directory")

    # Config directories:
    configs_description = {
        "features_config": "Config file for features extraction",
        "model_config": "Config file for the model",
        "data_config": "Config file for organize the data",
        "training_config": "Config file for the training hyperparams"
    }
    for config, description in configs_description.items():
        parser.add_argument(f"--{config}",help=description)
    
    # Parse args:
    args = vars(parser.parse_args())
    
    # Replace filenames with file content:
    for config in configs_description.keys():
        with open(args[config],"r") as f:
            args[config] = json.load(f)

    return args


def main():
    
    # Parse arguments:
    args = parse_args()

    # Tokenizer:
    tokenizer = init_tokenizer(args["tokenizer_dir"])

    # Features Extractor:
    features_extractor = init_features_extractor(tokenizer,**args["features_config"])

    # Main model:
    model = init_model(**args["model_config"])

    # Train and Validation Data:
    dataloaders = prepare_data_for_training(tokenizer,**args["data_config"])

    # Training
    results_history = train_classification_model(
        features_extractor=features_extractor,
        main_model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["validation"],
        **args["training_config"]
    )


if __name__ == "__main__":
    main()