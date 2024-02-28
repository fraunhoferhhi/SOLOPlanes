import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from config.solopmv_cfg import config as cfg
from datasets.scannet_datamodule import ScannetDataModule
from utils.util import set_seed
import datetime

def train(cfg, model, datamodule, find_lr = False, train_on_cluster=False):
    print(f"training {cfg.version}")
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    folder = cfg.version + year + month + day
    if train_on_cluster:
        folder = "solop_slurm"
    logger = TensorBoardLogger("lightning_logs_ckpts/", name=folder)

    torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        dirpath='lightning_logs_ckpts/solop_slurm',  
        filename='{epoch:02d}-{step}_{val_loss:.2f}',  
        save_top_k=1,  
        mode='min',  
        save_last=True,  
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        devices= "auto",
        accelerator="gpu",
        logger=logger,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[TQDMProgressBar(refresh_rate=20), EarlyStopping(monitor="val_loss", mode="min", patience=5), LearningRateMonitor(logging_interval='step'), checkpoint_callback]
    )

    if find_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        print(f"new lr {new_lr}")
        model.hparams.lr = new_lr

    trainer.fit(model, datamodule)



if __name__=="__main__":
    cfg = cfg_solop
    set_seed(cfg.seed)
    print(f"set seed {cfg.seed}")
    if cfg.model.path:
        print(f"loading model from {cfg.model.path}")
        model = cfg.model.init.load_from_checkpoint(cfg.model.path, cfg=cfg)
    elif cfg.model.init_pretrained_solo_path:
        print("initializing weights with pretrained solo...")
        model = cfg.model.init(cfg)
        pretrained_dict = torch.load(cfg.model.init_pretrained_solo_path)
        model_dict = model.state_dict()
        pretrained_state_dict = pretrained_dict['state_dict']

        upload_dict = {k:v for k,v in pretrained_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(upload_dict)
    else:
        print("training new model, initializing weights")
        model = cfg.model.init(cfg, init_weights=True)
    datamodule = ScannetDataModule(cfg)
    train(cfg, model, datamodule)

