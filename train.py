import sys
import time

import numpy as np
import torch
import torch.nn as nn

from util import remove_outliers
from model import seq_len, model, NUM_FEAT

epochs = 20
lr = 0.001
gamma = 0.95
batch_size = 20
eval_batch_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)


path = sys.argv[1]
print("dataset", path)
train_data = np.load(path)[:, :NUM_FEAT].astype(np.float32)

# normalize
train_data -= train_data.mean(axis=0, keepdims=True)
train_data /= train_data.std(axis=0, keepdims=True)


train_data = remove_outliers(train_data)


trivial_loss = np.mean((train_data[1:] - train_data[:-1]) ** 2)
print(f"Trivial loss (predicting no changes): {trivial_loss}")


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


train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


def get_batch(source, batch_index):
    data = source[batch_index : batch_index + seq_len]
    # Shift target by one step.
    target = source[batch_index + 1 : batch_index + 1 + seq_len]
    return data, target


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()

    batches = np.random.permutation(range(0, train_data.size(0) - seq_len, seq_len))
    for batch_counter, i in enumerate(batches):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()

        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        output = model(
            data,
            src_mask=src_mask,
        )

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
                    len(train_data) // seq_len,
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
        for i in range(0, data_source.size(0) - seq_len, seq_len):
            data, targets = get_batch(data_source, i)

            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

            output = eval_model(
                data,
                src_mask=src_mask,
            )

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
