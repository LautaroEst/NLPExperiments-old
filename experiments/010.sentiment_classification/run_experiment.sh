#! /bin/bash -ex

export PYTHONPATH=../../:$PYTHONPATH

# Entreno/cargo el tokenizer
python prepare_tokenizer.py \
    --config configs/tokenizer/$tokenizer.json \
    --out $out_dir/$(basename $tokenizer)







# Entrenamiento del clasificador
# python $scripts/train_classification_model.py \
#     --tokenizer_dir $out_dir/$(basename $tokenizer) \
#     --features_config configs/features/$features.json \
#     --model_config configs/model/$model.json \
#     --data_config configs/data/$data.json \
#     --training_config configs/training/$training.json
#     --out $out_dir/$(basename $tokenizer)/$features/$model/$data

echo "Finished OK"
