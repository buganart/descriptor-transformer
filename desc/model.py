import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class SampleModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters("config")
        descriptor_size = config.descriptor_size
        hidden_size = config.hidden_size
        num_layers = config.num_layers

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            descriptor_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, descriptor_size)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        batch_size = x.shape[0]
        h = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )
        x, _ = self.lstm(x, h)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        pred = output[:, -1, :].unsqueeze(1)

        loss = self.loss_function(pred, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def predict(self, data, step):
        all_descriptors = data
        batch_size, window_size, des_size = data.shape
        for i in range(step):
            input = all_descriptors[:, i:, :]
            # print("input", input)
            pred = self(input)
            new_descriptor = pred[:, 1, :].reshape(batch_size, 1, des_size)
            # print("new_descriptor", new_descriptor)
            all_descriptors = torch.cat((all_descriptors, new_descriptor), 1)
        return all_descriptors.detach().cpu().numpy()[:, -step:, :]
