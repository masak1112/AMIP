from common.utils import get_yaml
from data.datamodule import ClimateDataModule
from tqdm import tqdm
from datetime import timedelta
from data.amip_new import get_out_path
import h5py

config=get_yaml("configs/ae_SI_DDC.yaml")
datamodule = ClimateDataModule(config['data'])

train_dataset = datamodule.train_dataset

for i in tqdm(range(len(train_dataset))):
    data_datetime  = train_dataset.start_date + timedelta(hours=train_dataset.dates[i])
    data_year = data_datetime.year
    seconds_into_year = int(
        (data_datetime - train_dataset.datetime_class(data_year, 1, 1, hour=0,
                                                has_year_zero=train_dataset.has_year_zero)).total_seconds()
    )
    data_idx = seconds_into_year // 3600 // train_dataset.data_timedelta_hours
    data_file_path = get_out_path(train_dataset.data_dir, data_year, data_idx)

    try:
        f = h5py.File(data_file_path, 'r')
        f.close()
    except Exception as e:
        print(f"Error opening data file {data_file_path}: {e}")
        print(train_dataset.dates[i])
        print(i)
        break
            