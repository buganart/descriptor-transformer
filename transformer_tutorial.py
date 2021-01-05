import io
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from model import bptt, model, NUM_FEAT

epochs = 5  # The number of epochs

# url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
# test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
# tokenizer = get_tokenizer('basic_english')
# vocab = build_vocab_from_iterator(map(tokenizer,
#                                       iter(io.open(train_filepath,
#                                                    encoding="utf8"))))

# def data_process(raw_text_iter):
#   data = [torch.tensor([vocab[token] for token in tokenizer(item)],
#                        dtype=torch.long) for item in raw_text_iter]
#   return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# train_data = data_process(iter(io.open(train_filepath, encoding="utf8")))
# val_data = data_process(iter(io.open(valid_filepath, encoding="utf8")))
# test_data = data_process(iter(io.open(test_filepath, encoding="utf8")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)


# path = "data/ten-songs.npy"
path = sys.argv[1]
# "data/sine5.npy"
print("dataset", path)
train_data = np.load(path)[:, :NUM_FEAT].astype(np.float32)

# relative
# train_data = train_data[1:, :] - train_data[:-1, :]

# normalize
train_data -= train_data.mean(axis=0, keepdims=True)
train_data /= train_data.std(axis=0, keepdims=True)

# train_data = train_data[:20]

# n_use = 200_000
# train_data = train_data[:n_use]

all_data = torch.tensor(train_data).float().to(device)
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
    return data.to(device)


batch_size = 20
eval_batch_size = 100
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` function generates the input and target sequence for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we’d get the following two Variables for ``i`` = 0:
#
# .. image:: ../_static/img/transformer_input_target.png
#
# It should be noted that the chunks are along dimension 0, consistent
# with the ``S`` dimension in the Transformer model. The batch dimension
# ``N`` is along dimension 1.
#


def get_batch(source, batch_index):
    # seq_len = min(bptt, len(source) - 1 - batch_index)
    seq_len = bptt
    data = source[batch_index : batch_index + seq_len]
    # target = source[i+1:i+1+seq_len].reshape(-1)
    target = source[batch_index + 1 : batch_index + 1 + seq_len]
    return data, target
    # return data, data  # for transformer with encoder and decoder


######################################################################
# Run the model
# -------------
#

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
lr = 0.001  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)

import time


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()

    batches = np.random.permutation(range(0, train_data.size(0) - bptt, bptt))
    for batch_counter, i in enumerate(batches):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()

        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        # tgt_mask = model.generate_no_peak_mask(targets.size(0)).to(device)

        # tgt_mask = model.generate_square_subsequent_mask(targets.size(0)).to(device)
        # tgt_mask = model.generate_square_subsequent_mask(targets.size(0)).to(device)

        output = model(
            data,
            # targets,
            src_mask=src_mask,
            # tgt_mask=tgt_mask,
        )

        # loss = criterion(output.view(-1, ntokens), targets)
        # loss = criterion(output[:-1], targets[1:])  # for nn.Transfomer
        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 100
        if batch_counter % log_interval == 0 and batch_counter > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | {:5.6f}"
                "| ms/batch {:5.4f} | "
                "loss {:5.4f}".format(
                    epoch,
                    batch_counter,
                    len(train_data) // bptt,
                    scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss,
                )
            )
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for i in range(0, data_source.size(0) - bptt, bptt):
            data, targets = get_batch(data_source, i)
            # print("data", data.shape, "targets", targets.shape)

            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            # tgt_mask = model.generate_no_peak_mask(targets.size(0)).to(device)
            # tgt_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

            output = eval_model(
                data,
                # targets,
                src_mask=src_mask,
                # tgt_mask=tgt_mask,
            )

            # output_flat = output.view(-1, ntokens)
            # total_loss += len(data) * criterion(output_flat, targets).item()
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float("inf")
best_model = None

val_loss = evaluate(model, val_data)
print("-" * 89)
print("| start | valid loss {:5.4f} ".format(val_loss))

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print("-" * 89)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | "
        "".format(
            epoch,
            (time.time() - epoch_start_time),
            val_loss,
        )
    )
    print("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


######################################################################
# Evaluate the model with the test dataset
# -------------------------------------
#
# Apply the best model to check the result with the test dataset.

test_loss = evaluate(best_model, test_data)
print("=" * 89)
print(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, test_loss
    )
)
print("=" * 89)


torch.save(model.state_dict(), "model.pt")
