from nlp.features import CBOWFeatures

config = {
    
    "extractor_class": CBOWFeatures,
    "params": {
        "embeddings_dim": 300,
        "pretrained_file": None,
        "train_features": True
    }
}