import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Features, Value, ClassLabel
from transformers import PreTrainedTokenizerFast



def prepare_data_for_training(
        tokenizer: PreTrainedTokenizerFast,
        preprocess_dataset,
        data_collate,
        **data_config
    ):

    columns_to_return = ['input_ids', 'label', 'attention_mask']

    dataloaders = {}
    for split in ["train", "validation"]:
        dataset = load_dataset(**data_config[split]["loading_args"])
        dataset = dataset.map(
            lambda sample: preprocess_dataset(sample,tokenizer),
            **data_config[split]["mapping_args"]
        )
        dataset.set_format(type='torch', columns=columns_to_return)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=data_config[split]["batch_size"],
            shuffle=True,
            collate_fn=lambda batch: data_collate(batch,tokenizer)
        )

    return dataloaders