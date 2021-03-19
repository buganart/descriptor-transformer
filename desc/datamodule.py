import json
import tqdm
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import pytorch_lightning as pl


class DataModule_descriptor(pl.LightningDataModule):
    def __init__(self, config, isTrain=True, process_on_the_fly=True):
        super().__init__()
        self.data_path = Path(config.audio_db_dir)
        self.isTrain = isTrain
        self.process_on_the_fly = process_on_the_fly
        self.remove_outliers = config.remove_outliers

        self.window_size = config.window_size
        self.batch_size = config.batch_size

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
                    self.attribute_list = list(descriptor.keys())
                    self.attribute_list.remove("id")
                    self.attribute_list.remove("sample")
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
            if des_array.shape[0] == 0:
                return None, None, None

            # calculate interval and record last timestamp
            last_timestamp = sorted_data[-1][0]
            interval = sorted_data[-1][0] - sorted_data[-2][0]
        return des_array, last_timestamp, interval

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
        filepath_list = [
            path for path in filepath_list if Path(path).suffix in [".json", ".txt"]
        ]

        all_desc = []

        dataset_input = []

        test_input = []
        test_filename = []

        # process files in the filepath_list
        for path in tqdm.tqdm(filepath_list, desc="Descriptor Files"):
            des_array, last_timestamp, interval = self._load_descriptor_list(path)
            if des_array is None:
                continue
            # remove outliers
            if self.remove_outliers:
                des_array = self._remove_outliers_fn(des_array)
                # record all descriptor for statistics
            all_desc.append(des_array)
            num_des = des_array.shape[0]
            # print(f"{str(path.stem)[:20]} file number of descriptors:", num_des)
            if num_des <= window_size + 1:
                continue
            if self.isTrain:
                if self.process_on_the_fly:
                    continue
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
                # save metadata
                self.last_timestamp.append(last_timestamp)
                self.interval = interval

        if self.isTrain:
            # calculate mean and std
            # all_desc in shape (NUM_DESC, NUM_FEAT)
            all_descriptors = np.concatenate(all_desc, axis=0)
            self.dataset_mean = all_descriptors.mean(axis=0)
            self.dataset_std = all_descriptors.std(axis=0)

            if self.process_on_the_fly:
                # normalize data
                all_desc = [
                    (d - self.dataset_mean) / self.dataset_std for d in all_desc
                ]
                num_data_train = int(len(all_desc) * 0.9)
                self.dataset_input = all_desc[:num_data_train]
                self.dataset_val = all_desc[num_data_train:]
                print(
                    "train dataset shape (before process_on_the_fly):",
                    len(self.dataset_input),
                )
                print(
                    "val dataset shape (before process_on_the_fly):",
                    len(self.dataset_val),
                )
            else:
                # data input in shape (NUM_BATCH, WINDOW_SIZE, NUM_FEAT)
                dataset = np.concatenate(dataset_input, axis=0)
                # normalize data
                dataset = (dataset - self.dataset_mean) / self.dataset_std
                num_data_train = int(dataset.shape[0] * 0.9)
                self.dataset_input = dataset[:num_data_train]
                self.dataset_val = dataset[num_data_train:]
                print("train dataset shape:", self.dataset_input.shape)
                print("val dataset shape:", self.dataset_val.shape)
        else:
            self.test_input = np.concatenate(test_input, axis=0)
            self.test_filename = test_filename

    def train_dataloader(self):
        batch_size = self.batch_size
        if self.process_on_the_fly:
            dataset = RealTimeProcessDataset(self.dataset_input, self.window_size)
        else:
            dataset = TensorDataset(
                torch.tensor(self.dataset_input, dtype=torch.float32)
            )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        return dataloader

    def val_dataloader(self):
        batch_size = self.batch_size
        if self.process_on_the_fly:
            dataset = RealTimeProcessDataset(self.dataset_input, self.window_size)
        else:
            dataset = TensorDataset(torch.tensor(self.dataset_val, dtype=torch.float32))
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
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
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        return dataloader


#####
#   helper class
#####


class RealTimeProcessDataset(Dataset):
    def __init__(self, data_list, window_size, dtype=torch.float32):
        assert isinstance(data_list, list)
        self.data_list = data_list
        self.size = 64 * 1024
        self.window_size = window_size
        self.dtype = dtype

    def __getitem__(self, index):
        # index information is ignored
        sample_index = np.random.randint(0, len(self.data_list))

        sample = self.data_list[sample_index]
        sample_length = len(sample)
        window_index = np.random.randint(0, sample_length - self.window_size)
        item = sample[window_index : window_index + self.window_size]

        return torch.tensor(np.array(item)[np.newaxis, :, :], dtype=self.dtype)

    def __len__(self):
        return self.size

    def __add__(self, other):
        return RealTimeProcessDataset(self.data_list.append(other), self.window_size)
