#!/usr/bin/env ipython
from util import remove_outliers
import torch
import tqdm
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)

from util import remove_outliers
from model import model, seq_len, NUM_FEAT


# inputs = torch.ones([7, 2, 5])

# src_mask = model.generate_square_subsequent_mask(inputs.size(0))
# tgt_mask = model.generate_no_peak_mask(inputs.size(0))
# # tgt_mask = model.generate_square_subsequent_mask(targets.size(0)).to(device)
# # tgt_mask = model.generate_square_subsequent_mask(targets.size(0)).to(device)

# output = model(
#     inputs,
#     inputs,
#     src_mask=src_mask,
#     tgt_mask=tgt_mask,
# )

# print(output.shape)
model.load_state_dict(torch.load("model.pt"))
model.to(device)
model.eval()


def predict_step(model, inputs):
    # print("inputs", inputs.mean(axis=(0, 1)).detach().cpu().numpy())
    inputs = inputs.to(device)
    src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
    # tgt_mask = model.generate_no_peak_mask(inputs.size(0)).to(device)
    return model.forward(
        inputs,
        # inputs,
        src_mask=src_mask,
        # tgt_mask=tgt_mask,
    )


def predict(model, seq_len, steps, context):

    seq = context
    n_context = context.shape[0]

    for _ in tqdm.tqdm(range(steps)):

        inputs = seq[-seq_len:]
        with torch.no_grad():
            output = predict_step(model, inputs)

        # print("inputs", inputs)
        # print("output", output)
        # input()
        new_step = output[-1:].cpu()

        # add a bit of noise
        mag = output.cpu().std(axis=(0, 1)).reshape(1, 1, -1)
        new_step = new_step + torch.randn(*new_step.shape) * mag

        # print("inputs", inputs.cpu().numpy())
        # print("new_step", new_step.cpu().numpy())
        # print(input())
        seq = torch.cat([seq, new_step])
        # print("seq", seq.shape)

    return seq[n_context:].squeeze(axis=1)


# context = torch.ones([7, 2, 5])
# output = predict(model, 3, 5, context)
# print(output.shape)

import sys
import numpy as np

import sys

path = sys.argv[1]
train_data = np.load(path)[:, :NUM_FEAT].astype(np.float32)

train_data = remove_outliers(train_data)


# relative
# train_data = train_data[1:, :] - train_data[:-1, :]

# normalize

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)


def normalize(x):
    return (x - mean) / std


def unnormalize(x):
    return x * std + mean


train_data = normalize(train_data)
# train_data = train_data[:20]

# n_use = 200_000
# train_data = train_data[:n_use]

all_data = torch.tensor(train_data).float()
n_split = len(all_data) // 2

train_data = all_data[:n_split]
val_data = all_data[-n_split:]
test_data = train_data


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


# def predict(history, steps):

#     inputs = torch.tensor(
#         np.ones()


#     model.forward(
