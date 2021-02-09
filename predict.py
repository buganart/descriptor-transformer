#!/usr/bin/env ipython
import sys

import numpy as np
import torch
import tqdm

from model import NUM_FEAT, model, seq_len
from util import remove_outliers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)

model.load_state_dict(torch.load("model.pt"))
model.to(device)
model.eval()


def predict_step(model, inputs):
    inputs = inputs.to(device)
    src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
    return model.forward(
        inputs,
        src_mask=src_mask,
    )


def predict(model, seq_len, steps, context):

    seq = context
    n_context = context.shape[0]

    for _ in tqdm.tqdm(range(steps)):

        inputs = seq[-seq_len:]
        with torch.no_grad():
            output = predict_step(model, inputs)
        new_step = output[-1:].cpu()

        # XXX add a bit of noise to prevent totally flat predictions
        magnitude = output.cpu().std(axis=(0, 1)).reshape(1, 1, -1)
        new_step = new_step + torch.randn(*new_step.shape) * magnitude

        seq = torch.cat([seq, new_step])

    return seq[n_context:].squeeze(axis=1)


path = sys.argv[1]
train_data = np.load(path)[:, :NUM_FEAT].astype(np.float32)

train_data = remove_outliers(train_data)


mean = train_data.mean(axis=0)
std = train_data.std(axis=0)


def normalize(x):
    return (x - mean) / std


def unnormalize(x):
    return x * std + mean


train_data = normalize(train_data)

all_data = torch.tensor(train_data).float()
n_split = len(all_data) // 2

train_data = all_data[:n_split]
val_data = all_data[-n_split:]


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1, NUM_FEAT).transpose(0, 1).contiguous()
    return data


context = batchify(train_data[-seq_len:], 1)

pred = predict(model, seq_len, 2000, context)

train_data = unnormalize(train_data.detach().numpy())
val_data = unnormalize(val_data.detach().numpy())
pred = unnormalize(pred.detach().numpy())

np.save("val.npy", val_data)
np.save("context.npy", train_data)
np.save("pred.npy", pred)
