
from .TwoLayerNet import TwoLayerNet


def init_model(**model_config):
    if model_config["type"] == "TwoLayerNet":
        model = TwoLayerNet(**model_config["params"])
    else:
        raise ValueError(f"Model {model_config['type']} not supported")
    return model