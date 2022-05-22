import argparse
import json
import os
import torch

from nlp.tokenizers import init_tokenizer
from nlp.features import init_features_extractor

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


def main():
    config, tokenizer_dir, output_dir = parse_args()
    tokenizer = init_tokenizer(tokenizer_dir)
    features_extractor = init_features_extractor(tokenizer,**config)
    
    with open(os.path.join(output_dir,"features_config.json"),"w") as f:
        json.dump(config,f)
    torch.save({
        "state_dict": features_extractor.state_dict()
    },os.path.join(output_dir,"extractor_state_dict.pkl"))
    




if __name__ == "__main__":
    main()