from common.utils import get_yaml
from data.datamodule import ClimateDataModule
from tqdm import tqdm
from datetime import timedelta
from data.amip_new import get_out_path

config=get_yaml("configs/ae_SI_DDC.yaml")
datamodule = ClimateDataModule(config['data'])

train_dataset = datamodule.train_dataset

for i in tqdm(range(len(train_dataset))):
    try: 
        batch = train_dataset.__getitem__(i)
    except Exception as e:
        print(f"Error at index {i}: {e}")
        data_datetime  = train_dataset.start_date + timedelta(hours=train_dataset.dates[i])
        print(f"Time of error: {data_datetime}")
        data_year = data_datetime.year
        seconds_into_year = int(
            (data_datetime - train_dataset.datetime_class(data_year, 1, 1, hour=0,
                                                  has_year_zero=train_dataset.has_year_zero)).total_seconds()
        )
        data_idx = seconds_into_year // 3600 // train_dataset.data_timedelta_hours
        data_file_path = get_out_path(train_dataset.data_dir, data_year, data_idx)

        print(f"Data file path: {data_file_path}")
        print(data_datetime, data_year, seconds_into_year, data_idx)