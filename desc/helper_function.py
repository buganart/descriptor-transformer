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
        self.log_interval = log_interval
        self.save_model_path = Path(save_model_path)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if trainer.current_epoch % self.log_interval == 0:
            # log
            trainer.save_checkpoint(self.save_model_path)
            save_checkpoint_to_cloud(self.save_model_path)


# function to save/load files from wandb
def save_checkpoint_to_cloud(checkpoint_path):
    wandb.save(checkpoint_path)


def load_checkpoint_from_cloud(checkpoint_path="model_dict.pth"):
    checkpoint_file = wandb.restore(checkpoint_path)
    return checkpoint_file.name


####
# for prediction script


def save_descriptor_as_json(save_path, data, dataindex, datamodule, resume_run_id=None):
    save_path = Path(save_path)
    interval = int(datamodule.interval)
    attribute_list = datamodule.attribute_list

    num_data, prediction_length, _ = data.shape
    for i in range(num_data):
        data_index = dataindex[i]
        last_timestamp = int(datamodule.last_timestamp[data_index])
        current_timestamp = last_timestamp + interval
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
        # find data source name
        data_source_name = datamodule.test_filename[data_index]

        filename = str(data_source_name) + "_predicted_" + str(i) + ".txt"
        if resume_run_id:
            filename = str(resume_run_id) + "_" + filename
        json_path = save_path / filename
        with open(json_path, "w") as json_file:
            json.dump(stored, json_file)
