import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
import librosa


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
def wav2descriptor(filename, hop=1024, sr=44100):
    descriptors = []
    y, sr = librosa.load(filename, sr=sr)
    cent = np.ndarray.flatten(
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
    )
    flat = np.ndarray.flatten(librosa.feature.spectral_flatness(y=y, hop_length=hop))
    rolloff = np.ndarray.flatten(
        librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
    )
    rms = np.ndarray.flatten(librosa.feature.rms(y=y, hop_length=hop))
    f0 = np.ndarray.flatten(librosa.yin(y, 80, 10000, hop_length=hop))
    id = filename

    for x in range(cent.size):
        descriptors.append(
            {
                "cent": str(cent[x]),
                "flat": str(flat[x]),
                "rolloff": str(rolloff[x]),
                "rms": str(rms[x]),
                "f0": str(f0[x]),
                "_id": str(filename),
                "_sample": str(x * hop),
            }
        )
    return descriptors


def dir2descriptor(directory, hop=1024, sr=44100):
    directory = Path(directory)
    data = []
    for filename in directory.rglob("*.wav"):
        print("processing:", filename)
        descriptors = wav2descriptor(filename, hop=hop, sr=sr)
        data.append((str(filename), descriptors))
    return data


def save_json(savefile, data):
    savepath = Path(savefile.parent)
    savepath.mkdir(parents=True, exist_ok=True)
    savefile = savepath.absolute() / (str(savefile.stem) + str(savefile.suffix))
    with open(savefile, "w") as outfile:
        json.dump(data, outfile, indent=2)


def get_dataframe_from_json(filepath):
    filepath = Path(filepath)
    with open(filepath) as t:
        data = json.load(t)
        df = pd.DataFrame(data)

    for c in df.columns:
        if "_" not in c:
            df[c] = pd.to_numeric(df[c])

    return df


def save_descriptor_as_json(save_path, data, dataindex, datamodule, resume_run_id=None):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    attribute_list = datamodule.attribute_list

    num_data, prediction_length, _ = data.shape
    for i in range(num_data):
        data_index = dataindex[i]
        # for each data, save 1 json file
        stored = []
        for j in range(prediction_length):
            descriptor_values = {}
            for k in range(len(attribute_list)):
                descriptor_values[str(attribute_list[k])] = str(data[i, j, k])
            stored.append(descriptor_values)

        # save to json
        # find data source name
        data_source_name = datamodule.test_filename[data_index]

        filename = str(data_source_name) + "_prediction.json"
        if resume_run_id:
            filename = str(resume_run_id) + "_" + filename
        json_path = save_path / filename
        save_json(json_path, stored)
