from .cbow import CBOWFeatures


def init_features_extractor(tokenizer,**features_config):
    if features_config["type"] == "cbow":
        features_extractor = CBOWFeatures(tokenizer,**features_config["params"])
    else:
        raise ValueError(f"Features extractor of type {features_config['type']} not supported.")

    return features_extractor