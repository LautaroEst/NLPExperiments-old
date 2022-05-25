import torch
from torch import nn


class CBOWFeatures(nn.Module):
    
    def __init__(self,tokenizer,embeddings_dim=300,pretrained_file=None,train_features=True):
        super().__init__()
        embeddings = nn.Embedding(tokenizer.vocab_size,embeddings_dim,tokenizer.pad_token_id)
        self.embeddings = self._init_embeddings_from_file(pretrained_file,embeddings)

        if train_features:
            for param in self.embeddings.parameters():
                param.requires_grad = True
        else:
            for param in self.embeddings.parameters():
                param.requires_grad = False
        
    def _init_embeddings_from_file(self,pretrained_file,embeddings):
        if pretrained_file is None:
            return embeddings
        
        ## TO DO: Support pretrained embeddings
        return embeddings

    def forward(self,batch):
        cbow = self.embeddings(batch["input_ids"]).mean(axis=1) # Batch first
        return cbow