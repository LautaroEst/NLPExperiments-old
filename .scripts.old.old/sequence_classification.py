from importlib.machinery import SourceFileLoader
from types import ModuleType

from nlp.tokenization import load_pretrained_tokenizer, train_tokenizer


def import_configs_objs(config_file):
    # Dynamicaly loads the configuration file:
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    config_objs = vars(mod)
    return config_objs


def create_tokenizer(config_file,targets):
    # config_file = dependencies[0]
    config_objs = import_configs_objs(config_file)
    config = config_objs["tokenizer_config"]

    if config["is_pretrained"]:
        tokenizer = load_pretrained_tokenizer(config)
    else:
        prepare_batch_iterator = config_objs["prepare_batch_iterator"]
        batch_iterator, corpus_len = prepare_batch_iterator()
        tokenizer = train_tokenizer(config,batch_iterator,corpus_len)

    return {"tokenizer": tokenizer}


def get_features_extractor(tokenizer,config_file):
    return {"features": []}



def task_tokenizer():
    config_file = "/home/lestienne/NLPExperiments/experiments/01_sequence_classification/configs/tokenizer/wl_amazon_30000.py"
    return {
        "file_dep": [config_file],
        "targets": ["tokenizer.json"],
        "actions": [(create_tokenizer,[config_file])]
    }

# def task_features():
#     return {
#         "targets": ["features_extractor.pkl"],
#         "actions": [(get_features_extractor, )]
#     }

# def task_tokenizer():
#     return {
#         "actions": [(create_tokenizer,)],
#         "params": [{
#             "name": "config_file",
#             "long": "tokenizer_config",
#             "short": "t",
#             "default": "",
#             "type": str
#         }],
#         "verbosity": 2
#     }

