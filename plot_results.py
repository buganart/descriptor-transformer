#!/usr/bin/env python
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from model import NUM_FEAT

sns.set()

plt.rcParams["figure.figsize"] = (18, 18)
plt.rc("axes", labelsize=22)  # fontsize of the x and y labels

val = np.load("val.npy")
context = np.load("context.npy")
pred = np.load("pred.npy")


labels = "centroid flatness rolloff rms f0".split()
truth = pd.DataFrame(
    np.vstack([context, val]),
)
n_use = context.shape[0] + pred.shape[0]
truth = truth.iloc[n_use - 5_000 : n_use]

pred_index = np.arange(context.shape[0], context.shape[0] + pred.shape[0])
pred = pd.DataFrame(
    pred,
    index=pred_index,
)

fig, axs = plt.subplots(NUM_FEAT, 1)

if not hasattr(axs, "__iter__"):
    axs = [axs]


for feature, ax in enumerate(axs):
    truth[feature].plot(ax=ax, lw=1, linestyle="", marker=".", markersize=3)
    pred[feature].plot(ax=ax, lw=3, linestyle="", marker="x", color="r", markersize=3)
    ax.set_ylabel(labels[feature])

d = Path("plots")
d.mkdir(exist_ok=True)
path = d / f"pred_{datetime.utcnow().isoformat()}.png"
plt.savefig(path)
print(f"Figured saved to {path}")
plt.show()
