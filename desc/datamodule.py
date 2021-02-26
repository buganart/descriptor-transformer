import json
import tqdm
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pytorch_lightning as pl


class DataModule_descriptor(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_path = Path(config.audio_db_dir)
        self.attribute_list = None
        self.dataset_input = None
        self.dataset_target = None
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.remove_outliers = config.remove_outliers

    def remove_outliers_fn(self, x):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return np.array(x[(np.abs(x - mean) < std * 5).all(axis=1), :])

    # def prepare_data(self):
    #     pass
    def setup(self, stage=None):
        window_size = self.window_size
        filepath_list = self.data_path.rglob("*.*")
        # check files in filepath_list is supported (by extensions)
        filepath_list = [path for path in filepath_list if Path(path).suffix == ".txt"]

        attribute_list = []
        dataset_input = []
        dataset_target = []
        # process files in the filepath_list
        for path in tqdm.tqdm(filepath_list, desc="Descriptor Files"):

            with open(path) as json_file:
                data = json.load(json_file)
                data_list = []
                for des in data:
                    timestamp = next(iter(des))
                    descriptor = des[timestamp]
                    if len(attribute_list) == 0:
                        attribute_list = descriptor.keys()
                        attribute_list = sorted(attribute_list)
                    values = []
                    for k in attribute_list:
                        values.append(float(descriptor[k]))
                    data_list.append((int(timestamp), values))
                # sort value by timestamp
                sorted_data = sorted(data_list)
                # convert data into descriptor array
                des_array = [j for (i, j) in sorted_data]
                des_array = np.array(des_array)

                # normalize data and remove outliers
                des_array_mean = des_array.mean()
                des_array_std = des_array.std()
                des_array = (des_array - des_array_mean) / des_array_std
                if self.remove_outliers:
                    des_array = self.remove_outliers_fn(des_array)

                # pack descriptors into batches based on window_size
                num_des = des_array.shape[0]
                if num_des <= window_size + 1:
                    continue
                input_array = []
                target_array = []
                for i in range(num_des - window_size - 1):
                    input_batch = des_array[i : i + window_size]
                    target_batch = des_array[i + 1 : i + 1 + window_size]
                    input_array.append(input_batch)
                    target_array.append(target_batch)

                # add processed array to dataset
                dataset_input.append(input_array)
                dataset_target.append(target_array)

        self.attribute_list = attribute_list
        self.dataset_input = np.concatenate(dataset_input, axis=0)
        self.dataset_target = np.concatenate(dataset_target, axis=0)
        print("dataset_input", self.dataset_input.shape)

    def train_dataloader(self):
        batch_size = self.batch_size
        dataset = TensorDataset(
            torch.tensor(self.dataset_input, dtype=torch.float32),
            torch.tensor(self.dataset_target, dtype=torch.float32),
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )
        return dataloader
