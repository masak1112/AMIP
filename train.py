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
from modules.combined_module import CombinedModule
from data.datamodule import ClimateDataModule

# Lightning imports
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, WeightAveraging
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay=0.99):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # always update
        return True
    
def load_partial_weights(model, partial_ckpt):
    # Load only matching model weights (e.g. when swapping unpatchify head)
    ckpt = torch.load(partial_ckpt, map_location="cpu", weights_only=False)
    ckpt_state = ckpt["state_dict"]
    model_state = model.state_dict()
    # Filter to keys that exist in both and have matching shapes
    filtered = {k: v for k, v in ckpt_state.items()
                if k in model_state and v.shape == model_state[k].shape}
    skipped = [k for k in ckpt_state if k not in filtered]
    if skipped:
        print(f"Partial checkpoint: skipped {len(skipped)} keys with shape mismatch or missing:")
        for k in skipped:
            print(f"  {k}")
    model.load_state_dict(filtered, strict=False)
    print(f"Partial checkpoint: loaded {len(filtered)}/{len(ckpt_state)} keys from {partial_ckpt}")
    return model

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
    print(args.config)
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    trainconfig["checkpoint"] = args.checkpoint
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
    saving_checkpoint_path = os.path.split(args.checkpoint)[0]
    print("Saving checkpoints to: ", saving_checkpoint_path)


    os.makedirs(path, exist_ok=True) 
    print(f"Logging to: {path}")
    save_yaml(config, path + "config.yml")

    autoencoder = dataconfig.get("autoencoder", False)
    is_combined = modelconfig.get("model_name", "") == "Combined"

    datamodule = ClimateDataModule(dataconfig=dataconfig)

    if is_combined:
        # CombinedModule is evaluation-only and loads checkpoints internally.
        model = CombinedModule(config,
                               normalizer=datamodule.train_dataset)
    elif autoencoder:
        model = AutoencoderModule(config,
                                  normalizer=datamodule.train_dataset)
    else:
        model = TrainModule(config,
                            normalizer=datamodule.train_dataset)

    epoch_checkpoint = ModelCheckpoint(
        dirpath=saving_checkpoint_path,
        filename="model_{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
    )

    last_checkpoint = ModelCheckpoint(
        dirpath=saving_checkpoint_path,
        every_n_train_steps=100,
        save_last=True,
        save_top_k=0,
        enable_version_counter=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    debug_one_step_then_val = trainconfig.get("debug_one_step_then_val", False)
    max_epochs = 1 if debug_one_step_then_val else trainconfig["max_epochs"]
    limit_train_batches = 1 if debug_one_step_then_val else 1.0

    if debug_one_step_then_val:
        print("Debug mode enabled: running 1 training batch, then validation, then stopping.")

    trainer = L.Trainer(devices = trainconfig["devices"],
                        num_nodes = trainconfig.get("num_nodes", 1),
                        accelerator = trainconfig["accelerator"],
                        strategy = trainconfig["strategy"],
                        check_val_every_n_epoch = trainconfig["check_val_every_n_epoch"],
                        log_every_n_steps = trainconfig["log_every_n_steps"],
                        max_epochs = max_epochs,
                        limit_train_batches = limit_train_batches,
                        default_root_dir = path,
                        callbacks=[epoch_checkpoint, last_checkpoint, lr_monitor, EMAWeightAveraging(trainconfig["ema_decay"])],
                        logger=wandb_logger,
                        accumulate_grad_batches=trainconfig.get("accumulate_grad_batches", 1),
                        num_sanity_val_steps=trainconfig.get("num_sanity_val_steps", 1),
                        precision=trainconfig["precision"],)
    
    
    
    partial_ckpt = trainconfig.get("partial_checkpoint", None)

    if is_combined:
        # Weights are already loaded inside CombinedModule; don't pass ckpt_path.
        trainer.validate(model=model, datamodule=datamodule)
    elif partial_ckpt is not None:
        model = load_partial_weights(model, partial_ckpt)
        trainer.fit(model=model, datamodule=datamodule)
    elif trainconfig["checkpoint"] is not None:
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
    parser.add_argument('--checkpoint', default="/glade/work/bgong/amip/SI_checkpoints/last.ckpt", help='Path to the checkpoint to resume training')

    args = parser.parse_args()

    main(args)
