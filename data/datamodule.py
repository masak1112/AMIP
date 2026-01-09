import lightning as L
from torch.utils.data import DataLoader
import os 
from data.plasim import PLASIMData

class ClimateDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataconfig,) -> None:
        
        super().__init__()
        self.data_config = dataconfig
        self.dataset_config = dataconfig["dataset"]
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.normalizer_config = dataconfig["normalizer"]
        self.ae = dataconfig.get("ae", False)

        self.train_dataset = PLASIMData(data_path=self.dataset_config["train_data_path"],
                                        norm_stats_path=self.normalizer_config["norm_stats_path"],
                                        boundary_path=self.dataset_config["boundary_path"],
                                        time_path=self.dataset_config["train_times_path"],
                                        nsteps=self.dataset_config["training_nsteps"],   
                                        normalize_feature=True,
                                        ae = dataconfig["ae"],
                                        split='train')
        
        self.val_dataset = PLASIMData(data_path=self.dataset_config["val_data_path"],
                                        norm_stats_path=self.normalizer_config["norm_stats_path"],
                                        boundary_path=self.dataset_config["boundary_path"],
                                        time_path=self.dataset_config["val_times_path"],
                                        nsteps=self.dataset_config["val_nsteps"],   
                                        normalize_feature=True,
                                        ae = dataconfig["ae"],
                                        split="valid")
    
        self.normalizer = self.val_dataset.normalizer

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # Eager imports to avoid specific dependencies that are not needed in most cases

        if stage == "fit":
            pass 

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self, shuffle=True):
        self.pin_memory = False if self.num_workers == 0 else True
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=shuffle, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None