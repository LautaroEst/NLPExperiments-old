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


def create_tokenizer(config_file):
    config_objs = import_configs_objs(config_file)
    config = config_objs["tokenizer_config"]

    if config["is_pretrained"]:
        tokenizer = load_pretrained_tokenizer(config)
    else:
        prepare_batch_iterator = config_objs["prepare_batch_iterator"]
        batch_iterator, corpus_len = prepare_batch_iterator()
        tokenizer = train_tokenizer(config,batch_iterator,corpus_len)

    print(tokenizer)


def task_tokenizer():
    return {
        "actions": [(create_tokenizer,)],
        "params": [{
            "name": "config_file",
            "long": "tokenizer_config",
            "short": "t",
            "default": "",
            "type": str
        }],
        "verbosity": 2
    }

