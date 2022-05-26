




def preprocess_dataset(sample,tokenizer):
    encoded_input = tokenizer(
        sample["review_body"],
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return {
        "input_ids": encoded_input["input_ids"],
        "attention_mask": encoded_input["attention_mask"],
        "label": [label - 1 for label in sample["stars"]]
    }


config = {

    "train": {
        "loading_args": {
            "path": "amazon_reviews_multi",
            "name": "es",
            "split": "train"
        },
        "mapping_args": {
            "batched": True
        },
        "batch_size": 32
    },

    "validation": {
        "loading_args": {
            "path": "amazon_reviews_multi",
            "name": "es",
            "split": "validation"
        },
        "mapping_args": {
            "batched": True
        },
        "batch_size": 32
    }

}

