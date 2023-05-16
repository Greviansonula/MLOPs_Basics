import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

wandb_logger = WandbLogger(project="MLOPs Basics", log_model=True)

def main():
    print("init main")
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
        max_epochs=2,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1
    )
    # trainer.fit(cola_model, cola_data)
    trainer.fit(cola_model, cola_data)
    
    
if __name__ == "__main__":
    main()