
include config.mk

export PYTHONPATH:=../../:${PYTHONPATH}

TOKENIZER_CONFIG := configs/tokenizer/$(tokenizer).json
FEATURES_CONFIG := configs/features/$(features).json
DATA_CONFIG := configs/data/$(data).json
TRAINING_CONFIG := configs/data/$(training).json

TOKENIZER_OUT_DIR := $(out_dir)/$(shell basename $(tokenizer))
FEATURES_OUT_DIR := $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))
DATA_OUT_DIR := $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))/$(shell basename $(data))
TRAINING_OUT_DIR := $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))/$(shell basename $(training))

all: tokenizer features data training

tokenizer: $(TOKENIZER_OUT_DIR)
features: $(FEATURES_OUT_DIR)
data: $(DATA_OUT_DIR)
training: $(TRAINING_OUT_DIR)

$(out_dir):
	mkdir -p $(TRAINING_OUT_DIR)

$(TOKENIZER_OUT_DIR): $(TOKENIZER_CONFIG) $(out_dir)
	python prepare_tokenizer.py \
		--config $(TOKENIZER_CONFIG) \
		--out $(TOKENIZER_OUT_DIR)

$(FEATURES_OUT_DIR): $(FEATURES_CONFIG) $(TOKENIZER_OUT_DIR) $(shell find $(TOKENIZER_OUT_DIR) -maxdepth 1 -not -type d)
	python prepare_features.py \
		--config $(FEATURES_CONFIG) \
		--tokenizer_dir $(TOKENIZER_OUT_DIR) \
		--out $(FEATURES_OUT_DIR)

$(DATA_OUT_DIR): $(DATA_CONFIG) $(FEATURES_OUT_DIR) $(shell find $(TOKENIZER_OUT_DIR) -maxdepth 1 -not -type d) $(shell find $(FEATURES_OUT_DIR) -maxdepth 1 -not -type d)
	python prepare_data.py \
		--config $(DATA_CONFIG) \
		--tokenizer_dir $(TOKENIZER_OUT_DIR) \
		--out $(DATA_OUT_DIR)

$(TRAINING_OUT_DIR): $(TRAINING_CONFIG) $(FEATURES_OUT_DIR) $(shell find $(TOKENIZER_OUT_DIR) -maxdepth 1 -not -type d) $(shell find $(FEATURES_OUT_DIR) -maxdepth 1 -not -type d)
	python train_classifier.py \
		--config $(TRAINING_CONFIG) \
		--data_dir $(DATA_OUT_DIR) \
		--features_dir $(FEATURES_OUT_DIR) \
		--model_dir $(MODEL_OUT_DIR) \
		--out $(TRAINING_OUT_DIR)

clean: 
	rm -rf $(out_dir)