import lightning as L
from data.amip_new import get_data_loader

class ClimateDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataconfig,) -> None:
        
        super().__init__()
        self.data_config = dataconfig
        self.train_year_start = dataconfig['train_year_start']
        self.train_year_end = dataconfig['train_year_end']
        self.val_year_start = dataconfig['val_year_start']
        self.val_year_end = dataconfig['val_year_end']
        self.val_num_inferences = dataconfig['val_num_inferences']
        self.autoencoder = dataconfig.get("autoencoder", False)

        self.train_dataset, self.train_dataloader, _ = get_data_loader(dataconfig,
                                                                       distributed=True,
                                                                       year_start = self.train_year_start,
                                                                       year_end = self.train_year_end,
                                                                       num_inferences=0, # load entire dset
                                                                       train=True,
                                                                       validate=False)
        self.val_dataset, self.val_dataloader = get_data_loader(dataconfig,
                                                                distributed=True,
                                                                year_start = self.val_year_start,
                                                                year_end = self.val_year_end,
                                                                num_inferences=self.val_num_inferences, # load entire dset
                                                                train=False if not self.autoencoder else True, # for autoencoder, val is same as train
                                                                validate=True if not self.autoencoder else False)

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

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None