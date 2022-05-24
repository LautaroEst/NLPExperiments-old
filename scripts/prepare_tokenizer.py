

import argparse
import json


def parse_args():
    # Parser init
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--config",help="Json file with the tokenizer loading configuration")
    parser.add_argument("--output_dir",help="Output directory")
    args = vars(parser.parse_args())

    # Extract config and output directory
    
    output_dir = args["output_dir"]

    return config, output_dir


def main():
    config, output_dir = parse_args()
    print(config)
    print(output_dir)

if __name__ == "__main__":
    main()