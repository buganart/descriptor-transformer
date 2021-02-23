import numpy as np
import torch
import json
from pathlib import Path

from pytorch_lightning.callbacks.base import Callback
import pprint
import wandb


class SaveWandbCallback(Callback):
    def __init__(self, log_interval, save_model_path):
        super().__init__()
        self.epoch = 0
        self.log_interval = log_interval
        self.save_model_path = save_model_path

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if self.epoch % self.log_interval == 0:
            # log
            trainer.save_checkpoint(self.save_model_path)
            save_checkpoint_to_cloud(self.save_model_path)
        self.epoch += 1


# function to save/load files from wandb
def save_checkpoint_to_cloud(checkpoint_path):
    wandb.save(checkpoint_path)


def load_checkpoint_from_cloud(checkpoint_path="model_dict.pth"):
    checkpoint_file = wandb.restore(checkpoint_path)
    return checkpoint_file.name


####
# for prediction script
def load_test_data(config, test_data_path):
    test_data_path = Path(test_data_path)
    window_size = config.window_size

    attribute_list = []
    des_array = None

    with open(test_data_path) as json_file:
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
    #   cut according to the window size
    des_array = des_array[np.newaxis, -window_size:, :]
    #   also need to record attribute_list and timeframe for saving
    last_timestamp = sorted_data[-1][0]
    interval = sorted_data[-1][0] - sorted_data[-2][0]
    return torch.tensor(des_array, dtype=torch.float32), (
        int(last_timestamp),
        int(interval),
        attribute_list,
    )


def save_descriptor_as_json(save_path, data, audio_info, resume_run_id=None):
    save_path = Path(save_path)
    last_timestamp, interval, attribute_list = audio_info
    current_timestamp = last_timestamp + interval
    num_data, prediction_length, _ = data.shape
    for i in range(num_data):
        # for each data, save 1 json file
        stored = []
        for j in range(prediction_length):
            descriptor_info = {}
            descriptor_values = {}
            for k in range(len(attribute_list)):
                descriptor_values[str(attribute_list[k])] = str(data[i, j, k])
            descriptor_info[str(current_timestamp)] = descriptor_values
            stored.append(descriptor_info)
            # increment timestamp for next descriptor
            current_timestamp = current_timestamp + interval

        # save to json
        filename = "predicted_" + str(i) + ".txt"
        if resume_run_id:
            filename = str(resume_run_id) + "_" + filename
        json_path = save_path / filename
        with open(json_path, "w") as json_file:
            json.dump(stored, json_file)
