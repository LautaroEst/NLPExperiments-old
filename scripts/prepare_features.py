import argparse
import json
import os
import torch
import pickle

from nlp.tokenizers import load_tokenizer
from nlp.features import init_features_extractor
from nlp.utils import import_configs_objs

def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Json file with the tokenizer loading configuration")
    parser.add_argument("--tokenizer_dir",help="Directory that holds the tokenizer files")
    parser.add_argument("--out",help="Output directory")
    args = vars(parser.parse_args())

    # Process the arguments
    config_dict = import_configs_objs(args["config"])["config"]
    tokenizer_dir = args["tokenizer_dir"]
    output_dir = args["out"]

    return config_dict, tokenizer_dir, output_dir


def main():
    config, tokenizer_dir, output_dir = parse_args()
    tokenizer = load_tokenizer(tokenizer_dir)
    features_extractor = init_features_extractor(tokenizer,**config)

    with open(os.path.join(output_dir,"config.pkl"),"wb") as f:
        pickle.dump(config,f)
    torch.save(features_extractor.state_dict(),os.path.join(output_dir,"state_dict.pkl"))
    




if __name__ == "__main__":
    main()