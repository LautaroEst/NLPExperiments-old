import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Features, Value, ClassLabel
from transformers import PreTrainedTokenizerFast


def preprocess_dataset(sample,tokenizer):
    encoded_input = tokenizer(sample["review_body"],truncation=True,max_length=tokenizer.model_max_length)
    return {
        "input_ids": encoded_input["input_ids"],
        "attention_mask": encoded_input["attention_mask"],
        "label": [label - 1 for label in sample["stars"]]
    }
    


def prepare_data_for_training(tokenizer: PreTrainedTokenizerFast,**data_config):

    columns_to_return = ['input_ids', 'label', 'attention_mask']

    train_dataset = load_dataset("amazon_reviews_multi","es",split="train",cache_dir="../../data/",keep_in_memory=False)
    train_dataset = train_dataset.map(
        lambda sample: preprocess_dataset(sample,tokenizer),
        batched=True
    )
    print(train_dataset.column_names)
    train_dataset.set_format(type='torch', columns=columns_to_return)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: tokenizer.pad(batch,return_tensors="pt",max_length=tokenizer.model_max_length)
    )

    val_dataset = load_dataset("amazon_reviews_multi","es",split="validation",cache_dir="../../data/",keep_in_memory=False)
    val_dataset = val_dataset.map(
        lambda sample: preprocess_dataset(sample,tokenizer),
        batched=True
    )
    val_dataset.set_format(type='torch', columns=columns_to_return)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: tokenizer.pad(batch,return_tensors="pt",max_length=tokenizer.model_max_length)
    )
    return train_dataloader, val_dataloader