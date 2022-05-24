import argparse

from nlp.tokenizers import load_tokenizer
from nlp.features import load_features_extractor
from nlp.data import load_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config",help="Config file with hyperparams, etc.")
    parser.add_argument("--tokenizer_files_dir",help="The Tokenizer directory that holds all data to load the Tokenizer.")
    parser.add_argument("--features_files_dir",help="The features directory that holds all data to load the Features Extractor.")
    parser.add_argument("--data_files_dir",help="The data directory that holds all mapped and tokenized data.")
    parser.add_argument("--model_files_dir",help="The model directory that holds all data to load the model.")
    parser.add_argument("--output_dir",help="The directory to leave the training files (checkpoint, logs, etc.)")
    args = vars(parser.parse_args())

    config = args.pop("training_config")
    output_dir = args.pop("output_dir")
    directories = {key.split("_files_dir")[0]: dir_path for key, dir_path in args.items()}

    return config, output_dir, directories



def main():
    config, output_dir, directories = parse_args()

    tokenizer = load_tokenizer(directories["tokenizer"])
    features_extractor = load_features_extractor(tokenizer,directories["features"])
    dataloaders = load_dataloaders()





if __name__ == "__main__":
    main()