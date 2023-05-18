import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pandas import DataFrame
import wandb

import hydra
from omegaconf import OmegaConf

from data import DataModule
from model import ColaModel

wandb_logger = WandbLogger(project="MLOPs Basics", log_model=True)

class SampleVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        # get the predictions
        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        # predicted and labelled data
        df = DataFrame(
            {
                "Sentence":sentences, "Label": labels.numpy(), "Predicted": preds.numpy()
            }
        )

        # wrongly predicted data
        wrong_df = df[df["Label"] != df["Predicted"]]

        # Logging wrongly predicted dataframe as a table
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    cola_data = DataModule()
    cola_model = ColaModel()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )
    print("init trian")
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SampleVisualisationLogger(cola_data)],
        log_every_n_steps=10,
        deterministic=True,
    )
    # trainer.fit(cola_model, cola_data)
    trainer.fit(cola_model, cola_data)
    
    
if __name__ == "__main__":
    main()