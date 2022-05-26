import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl


class TrainedModel(pl.LightningModule):

    def __init__(self,features_extractor,main_model,train_features=True):
        super().__init__()
        self.features_extractor = features_extractor
        self.main_model = main_model

        if train_features:
            for param in features_extractor.parameters():
                param.requires_grad = True
        else:
            for param in features_extractor.parameters():
                param.requires_grad = False

    def forward(self,batch):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        y_hat = torch.argmax(F.log_softmax(scores),dim=-1)
        return y_hat

    def training_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        loss = F.cross_entropy(scores,batch["label"])
        self.log("train_loss",loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self,batch,batch_idx):
        features = self.features_extractor(batch)
        scores = self.main_model(features)
        loss = F.cross_entropy(scores,batch["label"])
        self.log("val_loss",loss)
        return loss

    def backward(self,loss,optimizer,optimizer_idx):
        loss.backward()

    def optimizer(self,epoch,batch_idx,optimizer,optimizer_idx):
        optimizer.step()
        

def train_sequence_classifier(
        features_extractor,main_model,train_dataloader,val_dataloader,**training_config
    ):

    train_features = training_config.pop("train_features",False)

    model = TrainedModel(features_extractor,main_model,train_features)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model,train_dataloader,val_dataloader)

    results_history = None
    return results_history