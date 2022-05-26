
def data_collate(batch,tokenizer):
    return tokenizer.pad(batch,return_tensors="pt",max_length=tokenizer.model_max_length)

config = {
    "train_batch_size": 32,
    "validation_batch_size": 32
}