# Default imports
import argparse
from datetime import datetime
import torch
from torch.optim.swa_utils import get_ema_avg_fn
import os 

# Custom imports
from common.utils import get_yaml, save_yaml
from modules.train_module import TrainModule
from modules.ae_module import AutoencoderModule
from data.datamodule import ClimateDataModule

# Lightning imports
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, WeightAveraging
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay=0.995):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # always update
        return True

def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(device) for device in args.devices]
    if args.seed is not None:
        trainconfig["seed"] = args.seed
    if args.wandb_mode is not None:
        trainconfig["wandb_mode"] = args.wandb_mode
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
    if args.description is not None:
        trainconfig["description"] = args.description
    
    return config, modelconfig, trainconfig, dataconfig

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    description = trainconfig.get("description", "")
    name = modelconfig["model_name"] + "_" + description + "_" + str(seed) + "_" + now
    wandb_logger = WandbLogger(project=trainconfig["project"],
                               name=name,
                               mode=trainconfig["wandb_mode"])
    path = trainconfig["log_dir"] + name + "/"
    config['training']["log_dir"] = path

    os.makedirs(path, exist_ok=True) 
    save_yaml(config, path + "config.yml")

    datamodule = ClimateDataModule(dataconfig=dataconfig)

    if "AE" in modelconfig["model_name"]:
        model = AutoencoderModule(config=config,
                                  normalizer=datamodule.train_dataset)
        monitor = "val/t2m"
        mode = 'min'
        every_n_train_steps = None
    else:
        model = TrainModule(config,
                            normalizer=datamodule.train_dataset)
        monitor = "step"
        mode = 'max'
        every_n_train_steps = 100

    checkpoint_callback  = ModelCheckpoint(
        monitor=monitor,
        filename= "model_{epoch:02d}_{step}_best",
        mode=mode,
        dirpath=path,
        save_last=True,
        save_top_k=1,
        every_n_train_steps=every_n_train_steps,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = L.Trainer(devices = trainconfig["devices"],
                        num_nodes = trainconfig.get("num_nodes", 1),
                        accelerator = trainconfig["accelerator"],
                        strategy = trainconfig["strategy"],
                        check_val_every_n_epoch = trainconfig["check_val_every_n_epoch"],
                        log_every_n_steps = trainconfig["log_every_n_steps"],
                        max_epochs = trainconfig["max_epochs"],
                        default_root_dir = path,
                        callbacks=[checkpoint_callback, lr_monitor, EMAWeightAveraging(trainconfig["ema_decay"])],
                        logger=wandb_logger,
                        accumulate_grad_batches=trainconfig.get("accumulate_grad_batches", 1),
                        num_sanity_val_steps=trainconfig.get("num_sanity_val_steps", 1),
                        precision=trainconfig["precision"],)
    
    if trainconfig["checkpoint"] is not None:
        trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path=trainconfig["checkpoint"],
                weights_only=False)
    else:
        trainer.fit(model=model, 
                datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Path to the checkpoint to resume training')
    args = parser.parse_args()

    main(args)