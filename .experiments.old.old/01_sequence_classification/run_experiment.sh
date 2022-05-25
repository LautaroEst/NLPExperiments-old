#!/bin/bash -ex

export PYTHONPATH=../../:$PYTHONPATH
scripts=../../scripts

## CONFIGURATION
tokenizer_config_file=configs/tokenizer/wl_amazon_30000.py
features_config_file=configs/features/cbow300.py
## END CONFIGURATION

# doit -f $scripts/sequence_classification.py tokenizer --tokenizer_config $(pwd)/$tokenizer_config_file
doit -f $scripts/sequence_classification.py tokenizer