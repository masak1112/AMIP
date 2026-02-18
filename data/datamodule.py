import lightning as L
from torch.utils.data import DataLoader
from data.amip import AMIPData, ClimatologyData

class ClimateDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataconfig,) -> None:
        
        super().__init__()
        self.data_config = dataconfig
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.norm_stats_path = dataconfig['norm_stats_path']
        self.use_climatology = dataconfig.get('use_climatology', False)
        self.normalize = dataconfig.get('normalize', True)

        self.train_dataset = AMIPData(data_path=dataconfig["train_data_path"],
                                        norm_stats_path=self.norm_stats_path,
                                        nsteps=dataconfig["training_nsteps"],  
                                        split='test',
                                        downsample_levels=dataconfig.get("downsample_levels", 1),
                                        normalize=self.normalize)
        
        self.val_dataset = AMIPData(data_path=dataconfig["val_data_path"],
                                        norm_stats_path=self.norm_stats_path,
                                        nsteps=dataconfig["val_nsteps"],  
                                        split='valid',
                                        horizon=dataconfig.get("val_horizon", -1),
                                        downsample_levels=dataconfig.get("downsample_levels", 1),
                                        normalize=self.normalize)
        if self.use_climatology:
            self.climatology_dataset = ClimatologyData(data_path=dataconfig["train_data_path"],
                                                    norm_stats_path=self.norm_stats_path,
                                                    climatology_path=dataconfig["climatology_path"],
                                                    horizon=dataconfig["climatology_horizon"],
                                                    start_time=dataconfig["climatology_start"])
    
        self.normalizer = self.train_dataset.n

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
        weather_dataloader = DataLoader(self.val_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        num_workers=self.num_workers,)
        
        if self.use_climatology:
            climatology_dataloser = DataLoader(self.climatology_dataset, 
                                            batch_size=1, 
                                            shuffle=False, 
                                            num_workers=self.num_workers,)
            
            return [weather_dataloader, climatology_dataloser]
        else:
            return weather_dataloader

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None