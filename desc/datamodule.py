import json
import tqdm
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pytorch_lightning as pl


class DataModule_descriptor(pl.LightningDataModule):
    def __init__(self, config, isTrain=True):
        super().__init__()
        self.data_path = Path(config.audio_db_dir)
        self.isTrain = isTrain
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.remove_outliers = config.remove_outliers

        self.dataset_input = None
        self.dataset_target = None

        self.attribute_list = []
        self.dataset_mean = None
        self.dataset_std = None

        #
        self.test_input = None
        self.test_filename = None
        self.interval = -1
        self.last_timestamp = []

    def _remove_outliers_fn(self, x):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return np.array(x[(np.abs(x - mean) < std * 5).all(axis=1), :])

    # def prepare_data(self):
    #     pass

    def _load_descriptor_list(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
            data_list = []
            for des in data:
                timestamp = next(iter(des))
                descriptor = des[timestamp]
                if len(self.attribute_list) == 0:
                    self.attribute_list = descriptor.keys()
                    self.attribute_list = sorted(self.attribute_list)
                values = []
                for k in self.attribute_list:
                    values.append(float(descriptor[k]))
                data_list.append((int(timestamp), values))
            # sort value by timestamp
            sorted_data = sorted(data_list)
            # convert data into descriptor array
            des_array = [j for (i, j) in sorted_data]
            des_array = np.array(des_array)

            # calculate interval and record last timestamp
            last_timestamp = sorted_data[-1][0]
            self.last_timestamp.append(last_timestamp)
            self.interval = sorted_data[-1][0] - sorted_data[-2][0]
        return des_array

    def _descriptor_batchify(self, des_array, window_size):
        # pack descriptors into batches based on window_size
        num_des = des_array.shape[0]
        input_array = []
        for i in range(num_des - window_size):
            input_batch = des_array[i : i + window_size]
            input_array.append(input_batch)
        return input_array

    def setup(self, stage=None):
        window_size = self.window_size
        filepath_list = self.data_path.rglob("*.*")
        # check files in filepath_list is supported (by extensions)
        filepath_list = [path for path in filepath_list if Path(path).suffix == ".txt"]

        all_desc = []

        dataset_input = []

        test_input = []
        test_filename = []

        # process files in the filepath_list
        for path in tqdm.tqdm(filepath_list, desc="Descriptor Files"):
            des_array = self._load_descriptor_list(path)
            # remove outliers
            if self.remove_outliers:
                des_array = self._remove_outliers_fn(des_array)
                # record all descriptor for statistics
            all_desc.append(des_array)
            num_des = des_array.shape[0]

            if num_des <= window_size + 1:
                continue
            if self.isTrain:
                # process as train data
                input_array = self._descriptor_batchify(des_array, window_size)
                # add processed array to dataset
                dataset_input.append(input_array)
            else:
                # process data for prediction
                last_descriptors = des_array[np.newaxis, :window_size, :]
                test_input.append(last_descriptors)
                # also record filename, trim to 20 chars
                test_filename.append(str(path.stem)[:20])

        if self.isTrain:
            # calculate mean and std
            # all_desc in shape (NUM_DESC, NUM_FEAT)
            all_desc = np.concatenate(all_desc, axis=0)
            self.dataset_mean = all_desc.mean(axis=0)
            self.dataset_std = all_desc.std(axis=0)
            # data input in shape (NUM_BATCH, WINDOW_SIZE, NUM_FEAT)
            self.dataset_input = np.concatenate(dataset_input, axis=0)
            print("dataset shape:", self.dataset_input.shape)
            # normalize data
            self.dataset_input = (
                self.dataset_input - self.dataset_mean
            ) / self.dataset_std
        else:
            self.test_input = np.concatenate(test_input, axis=0)
            self.test_filename = test_filename

    def train_dataloader(self):
        batch_size = self.batch_size
        dataset = TensorDataset(torch.tensor(self.dataset_input, dtype=torch.float32))
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )
        return dataloader

    #
    def test_dataloader(self):
        batch_size = self.batch_size
        dataset = TensorDataset(
            torch.tensor(
                (self.test_input - self.dataset_mean) / self.dataset_std,
                dtype=torch.float32,
            ),
            torch.tensor(range(len(self.test_filename))),
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )
        return dataloader
