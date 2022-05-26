import os
import json
import pickle

from .cbow import CBOWFeatures

import torch


def init_features_extractor(tokenizer,**features_config):
    if features_config["extractor_class"] == CBOWFeatures:
        features_extractor = CBOWFeatures(tokenizer,**features_config["params"])
    else:
        raise ValueError(f"Features extractor of type {features_config['type']} not supported.")

    return features_extractor


def load_features_extractor(tokenizer,features_dir):
    
    # Load the config.json file to init the features extractor
    with open(os.path.join(features_dir,"config.pkl"),"rb") as f:
        config = pickle.load(f)
    
    # Default initialization of the extractor
    if config["extractor_class"] == CBOWFeatures:
        extractor = CBOWFeatures(tokenizer,**config["params"])
    else:
        raise ValueError(f"Features extractor of type {config['extractor_class']} not supported.")

    # Load the state dictionary
    state_dict = torch.load(os.path.join(features_dir,"state_dict.pkl"))
    extractor.load_state_dict(state_dict)

    return extractor