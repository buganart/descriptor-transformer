#!/usr/bin/env pyhton
import copy
from pathlib import Path
import warnings
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
from pytorch_forecasting.data.examples import get_stallion_data

n_var = 1

fname = sys.argv[1]
data = np.load(fname)
variables = "centroid flatness rolloff rms f0".split()
data = pd.DataFrame(data, columns=variables)

data = data.iloc[:, :n_var]
variables = variables[:n_var]

data["time_idx"] = np.arange(len(data))
data["constant"] = 1

max_prediction_length = 10
max_encoder_length = 20
# training_cutoff = data["time_idx"].max() - max_prediction_length
training_cutoff = int(len(data) * 0.8)

training = TimeSeriesDataSet(
    data.iloc[:training_cutoff],
    time_idx="time_idx",
    #
    # target=variables,
    target=variables[0],  # XXX
    #
    group_ids=["constant"],
    min_encoder_length=max_encoder_length
    // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    # static_categoricals=["agency", "sku"],
    # static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    # time_varying_known_categoricals=["special_days", "month"],
    # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=variables,
    # target_normalizer=GroupNormalizer(
    #     groups=["agency", "sku"], transformation="softplus"
    # ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=4
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=4
)

# for x, (y, _) in iter(val_dataloader):
#     print("y", type(y), y)
#     sys.exit()

actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
# actuals = torch.cat([torch.cat(y) for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
baseline = (actuals - baseline_predictions).abs().mean().item()
print("baseline", baseline)

pl.seed_everything(42)
trainer = pl.Trainer(
    gpus=1,
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)
print(trainer)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.001,
    hidden_size=4,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)

# print(tft)

trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = tft.predict(val_dataloader)
error = (actuals - predictions).abs().mean()
print("error", error)
