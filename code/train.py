from reloo_model import ReLooModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob
from omegaconf import OmegaConf

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=1800,
        check_val_every_n_epoch=50,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    model = ReLooModel(opt)
    trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
    validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    if opt.model.is_continue and len(checkpoints) > 0:
        trainer.fit(model, trainset, validset, ckpt_path=checkpoints[-1])
    else: 
        OmegaConf.save(opt, "config.yaml")
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    main()