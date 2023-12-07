import torch
import imageio

import wandb
import pytorch_lightning as pl
from diffusion.model import DiffusionModel
from diffusion.data import get_dataloader
from pytorch_lightning.loggers import WandbLogger


def train(opt):
    """_summary_

    Args:
        opt (_type_): _description_
    """
    # Get data
    train_loader, val_loader, info_dict = get_dataloader(opt.dataset_choice, opt.batch_size)

    # Get model

    # Train
    wandb_logger = WandbLogger(log_model="all")

    trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs = 2, # to test
    log_every_n_steps=10,
    accelerator="auto",
    enable_checkpointing=True
    )
    
    if not bool(opt.model_path):
        model = DiffusionModel(info_dict['size']*info_dict['size'], opt.diffusion_steps, info_dict['depth'])
    else:
        model = DiffusionModel.load_from_checkpoint(opt.model_path,info_dict['size']*info_dict['size'], opt.diffusion_steps, info_dict['depth'])

    trainer.fit(model, train_loader, val_loader)
    print("Completed!")