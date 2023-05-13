import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ColaModel

def main():
    print("init main")
    cola_data = DataModule()
    cola_model = ColaModel()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="train_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=3, verbose=True, mode="min"
    )
    print("init trian")
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=2,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)
    
    
if __name__ == "__main__":
    main()