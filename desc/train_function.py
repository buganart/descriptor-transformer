import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pathlib import Path
from argparse import Namespace
import json

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import WandbLogger
import pprint
import wandb

from desc.datamodule import DataModule_descriptor
from desc.model import SimpleRNNModel, TransformerEncoderOnlyModel
from desc.helper_function import (
    SaveWandbCallback,
    save_checkpoint_to_cloud,
    load_checkpoint_from_cloud,
    save_descriptor_as_json,
)


def _get_models(model_name):
    if model_name == "RNN":
        MODEL_CLASS = SimpleRNNModel
    else:
        MODEL_CLASS = TransformerEncoderOnlyModel
    return MODEL_CLASS


def save_model_args(config, run):
    filepath = str(Path(run.dir).absolute() / "model_args.json")

    config = vars(config)
    config_dict = {}
    for k in config.keys():
        config_dict[k] = str(config[k])
    with open(filepath, "w") as fp:
        json.dump(config_dict, fp)
    save_checkpoint_to_cloud(filepath)


def get_resume_run_config(resume_id):
    # all config will be replaced by the stored one in wandb
    api = wandb.Api()
    previous_run = api.run(f"demiurge/descriptor_model/{resume_id}")
    config = Namespace(**previous_run.config)
    return config


def init_wandb_run(config, run_dir="./", mode="run"):
    resume_run_id = config.resume_run_id
    entity = "demiurge"
    run_dir = Path(run_dir).absolute()

    if resume_run_id:
        run_id = resume_run_id
    else:
        run_id = wandb.util.generate_id()

    run = wandb.init(
        project="descriptor_model",
        id=run_id,
        entity=entity,
        resume=True,
        dir=run_dir,
        mode=mode,
    )

    print("run id: " + str(wandb.run.id))
    print("run name: " + str(wandb.run.name))
    wandb.watch_called = False
    # run.tags = run.tags + (selected_model,)
    return run


def setup_datamodule(config, run, isTrain=True):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataModule = DataModule_descriptor(config, isTrain)
    dataModule.setup()
    if isTrain:
        # save mean std to npz
        data_stat_path = str(Path(run.dir).absolute() / "data_stat.npz")
        np.savez(
            data_stat_path,
            mean=dataModule.dataset_mean,
            std=dataModule.dataset_std,
        )
        save_checkpoint_to_cloud(data_stat_path)
    else:
        # load mean std from npz and write to datamodule
        load_checkpoint_from_cloud(checkpoint_path="data_stat.npz")
        data_stat_path = str(Path(run.dir).absolute() / "data_stat.npz")
        dataFile = np.load(data_stat_path)
        mean = dataFile["mean"]
        std = dataFile["std"]
        dataModule.dataset_mean = mean
        dataModule.dataset_std = std
    return dataModule


def setup_model(config, run):
    selected_model = config.selected_model
    # model
    MODEL_CLASS = _get_models(config.selected_model)

    if config.resume_run_id:
        checkpoint_path = str(Path(run.dir).absolute() / "checkpoint.ckpt")
        checkpoint_prev_path = str(Path(run.dir).absolute() / "checkpoint_prev.ckpt")
        new_ckpt_loaded = False
        try:
            # Download file from the wandb cloud.
            load_checkpoint_from_cloud(checkpoint_path="checkpoint.ckpt")
            extra_trainer_args = {"resume_from_checkpoint": checkpoint_path}
            model = MODEL_CLASS.load_from_checkpoint(checkpoint_path)
            new_ckpt_loaded = True
        except:
            # Download previous successfully loaded checkpoint file
            load_checkpoint_from_cloud(checkpoint_path="checkpoint_prev.ckpt")
            extra_trainer_args = {"resume_from_checkpoint": checkpoint_prev_path}
            model = MODEL_CLASS.load_from_checkpoint(checkpoint_prev_path)

        if new_ckpt_loaded:
            print(
                "checkpoint loaded. Save a copy of successfully loaded checkpoint to the cloud."
            )
            # save successfully loaded checkpoint file as checkpoint_prev.ckpt
            os.rename(checkpoint_path, checkpoint_prev_path)
            save_checkpoint_to_cloud(checkpoint_prev_path)
    else:
        extra_trainer_args = {}
        model = MODEL_CLASS(config)

    return model, extra_trainer_args


def train(config, run, model, dataModule, extra_trainer_args):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # wandb logger setup
    wandb_logger = WandbLogger(
        experiment=run, log_model=True, save_dir=Path(run.dir).absolute()
    )

    # log config
    wandb.config.update(config)
    save_model_args(config, run)
    pprint.pprint(vars(config))

    checkpoint_path = str(Path(run.dir).absolute() / "checkpoint.ckpt")
    callbacks = [SaveWandbCallback(config.save_interval, checkpoint_path)]

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=wandb.run.dir,
        checkpoint_callback=None,
        **extra_trainer_args,
    )

    # train
    trainer.fit(model, dataModule)


# sample script
def main():
    data_location = "../tests/"
    config_dict = dict(
        audio_db_dir=data_location,
        experiment_dir="../",
        resume_run_id="",
        window_size=10,
        learning_rate=1e-4,
        batch_size=16,
        epochs=3,
        save_interval=2,
        notes="",
        hidden_size=100,
        num_layers=3,
        remove_outliers=True,
        selected_model="RNN",
        descriptor_size=5,
        dim_pos_encoding=50,
        nhead=5,
        num_encoder_layers=1,
        dropout=0.1,
        positional_encoding_dropout=0,
        dim_feedforward=128,
    )
    config = Namespace(**config_dict)
    config.seed = 1234

    # run offline
    os.environ["WANDB_MODE"] = "dryrun"

    run = init_wandb_run(config, run_dir=config.experiment_dir, mode="offline")
    datamodule = setup_datamodule(config, run)
    model, extra_trainer_args = setup_model(config, run)
    train(config, run, model, datamodule, extra_trainer_args)

    #########
    # predict
    #########
    # config = get_resume_run_config(resume_run_id)
    # config.resume_run_id = resume_run_id
    # run = init_wandb_run(config, run_dir="./", mode="offline")
    # model, _ = setup_model(config, run)
    model.eval()
    # # construct test_data

    # testdatamodule = setup_datamodule(config, run, isTrain=False)
    # test_dataloader = testdatamodule.test_dataloader(
    #     datamodule.dataset_mean, datamodule.dataset_std
    # )
    # test_data, filenames = next(iter(test_dataloader))
    test_dataloader = datamodule.train_dataloader()
    test_data = next(iter(test_dataloader))[0]
    pred = model.predict(test_data, 5)
    print("pred.shape", pred.shape)
    # save_descriptor_as_json(data_location, prediction, audio_info)


if __name__ == "__main__":
    main()
