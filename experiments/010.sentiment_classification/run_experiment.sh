#! /bin/bash -ex

export PYTHONPATH=../../:$PYTHONPATH
scripts=../../scripts

out_dir=output
tokenizer=from_scratch/wordlevel.amazon.30000
features=cbow300train
model=two_layer_net.300.100.5
data=amazon
training=lr1e4.adam.epochs5

# Preparo la carpeta de salida
mkdir -p $out_dir/$(basename $tokenizer)/$features/$model/$data

# Entreno/cargo el tokenizer
# python $scripts/prepare_tokenizer.py --config configs/tokenizer/$tokenizer.json --out $out_dir/$(basename $tokenizer)

# Entrenamiento del clasificador
python $scripts/train_classification_model.py \
    --tokenizer_dir $out_dir/$(basename $tokenizer) \
    --features_config configs/features/$features.json \
    --model_config configs/model/$model.json \
    --data_config configs/data/$data.json \
    --training_config configs/training/$training.json
    --out $out_dir/$(basename $tokenizer)/$features/$model/$data

echo "Finished OK"
