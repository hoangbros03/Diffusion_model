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
    train_loader, val_loader, info_dict = get_dataloader(opt.model_conf.dataset_choice, opt.model_conf.batch_size)

    # Get model

    # Train
    if bool(opt.log.wandb_key):
        wandb_logger = WandbLogger(log_model="all")

        trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs = opt.model_conf.max_epochs, # to test
        log_every_n_steps=10,
        accelerator="auto",
        enable_checkpointing=True
        )
    else:
        trainer = pl.Trainer(
        max_epochs = opt.model_conf.max_epochs, # to test
        log_every_n_steps=10,
        accelerator="auto",
        enable_checkpointing=True
        )
    
    if not bool(opt.model_conf.model_path):
        model = DiffusionModel(info_dict['size']*info_dict['size'], opt.model_conf.diffusion_steps, info_dict['depth'])
    else:
        model = DiffusionModel.load_from_checkpoint(opt.model_conf.model_path,info_dict['size']*info_dict['size'], opt.model_conf.diffusion_steps, info_dict['depth'])

    trainer.fit(model, train_loader, val_loader)
    print("Completed!")