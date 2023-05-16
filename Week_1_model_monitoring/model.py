import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F 
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import torchmetrics
from sklearn.metrics import confusion_matrix

class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()
        
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.num_classes = 2
        
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(task="binary", num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            task="binary", average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            task="binary", average="macro", num_classes=self.num_classes
        )

        self.precision_micro_metric = torchmetrics.Precision(task="binary", average="micro")
        self.recall_micro_metric = torchmetrics.Recall(task="binary", average="micro")

        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
    
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch['label']
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def on_validation_epoch_end(self):
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        preds = torch.argmax(logits, 1)

        cm = confusion_matrix(labels.numpy(), preds.numpy())

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])