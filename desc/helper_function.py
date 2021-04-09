import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
import librosa
import shutil


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
    filename = Path(filename)
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

    num_descriptors = cent.shape[0]

    id = [str(filename)] * num_descriptors
    sample = np.arange(num_descriptors) * hop

    descriptors = {
        "cent": cent.tolist(),
        "flat": flat.tolist(),
        "rolloff": rolloff.tolist(),
        "rms": rms.tolist(),
        "f0": f0.tolist(),
        "_id": id,
        "_sample": sample.tolist(),
    }

    return descriptors


def process_descriptors(data_path, hop_length, sr):
    # find .wav files
    data_path = Path(data_path)
    wav_list = data_path.rglob("*.wav")

    descriptor_list = [
        path.stem
        for path in data_path.rglob("*.*")
        if Path(path).suffix in [".json", ".txt"]
    ]

    processed_data_path = data_path / "processed_descriptors"
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # process wav not being processed (not in descriptor_list) to processed_data_path
    for wav_path in wav_list:
        if wav_path.stem not in descriptor_list:
            descriptors = wav2descriptor(wav_path, hop=hop_length, sr=sr)
            num_descriptors = len(descriptors["cent"])
            print(f"processed {wav_path}. number of descriptors: {num_descriptors}")
            savefile = processed_data_path / (str(wav_path.stem) + ".json")
            save_json(savefile, descriptors)

    # copy processed descriptors (those in descriptor_list) to processed_data_path
    descriptor_list = [
        path
        for path in data_path.rglob("*.*")
        if Path(path).suffix in [".json", ".txt"]
    ]
    for file in descriptor_list:
        if not file.parent.samefile(processed_data_path):
            print(f'copied file "{file}" to "{processed_data_path}"')
            shutil.copy(file, processed_data_path)

    return processed_data_path


def dir2descriptor(directory, hop=1024, sr=44100):
    directory = Path(directory)
    processed_data_path = process_descriptors(directory, hop, sr)
    data = []
    descriptor_list = [
        path
        for path in processed_data_path.rglob("*.*")
        if Path(path).suffix in [".json", ".txt"]
    ]
    for filename in descriptor_list:
        filename = Path(filename)
        with open(filename) as t:
            descriptors = json.load(t)
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
        descriptors_value = {}
        for k in range(len(attribute_list)):
            attribute_name = str(attribute_list[k])
            descriptors_value[attribute_name] = data[i, :, k].tolist()

        # save to json
        # find data source name
        data_source_name = datamodule.test_filename[data_index]

        filename = str(data_source_name) + "_prediction.json"
        if resume_run_id:
            filename = str(resume_run_id) + "_" + filename
        json_path = save_path / filename
        save_json(json_path, descriptors_value)
