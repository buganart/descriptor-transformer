import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import wandb


########################
#   model
########################


class SimpleLSTMModel(pl.LightningModule):
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

        self.model_ep_loss_list = []

    def on_train_epoch_end(self, epoch_output=None):
        log_dict = {"epoch": self.current_epoch}

        loss = np.mean(self.model_ep_loss_list)
        log_dict["loss"] = loss

        wandb.log(log_dict, step=self.current_epoch)
        self.model_ep_loss_list = []

    def validation_epoch_end(self, validation_step_outputs):
        outputs = []
        for pred in validation_step_outputs:
            outputs.append(pred.detach().cpu().numpy())
        wandb.log({"val_loss": np.mean(outputs)}, step=self.current_epoch)

    def forward(self, x, hidden=None):
        batch_size, seq_len, d_size = x.shape
        if hidden is not None:
            h = hidden
        else:
            h = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size).type_as(x),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).type_as(x),
            )
        out, h2 = self.lstm(x, h)
        out = out.reshape((batch_size * seq_len, -1))
        out = self.linear(out)
        out = out.reshape((batch_size, seq_len, d_size))
        if hidden is not None:
            return out, h2
        else:
            return out

    def training_step(self, batch, batch_idx):
        data = batch[0]
        data, target = data[:, :-1, :], data[:, 1:, :]
        pred = self(data)

        loss = self.loss_function(pred, target)

        self.model_ep_loss_list.append(loss.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        data, target = data[:, :-1, :], data[:, 1:, :]
        pred = self(data)

        loss = self.loss_function(pred, target)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def predict(self, data, step):
        batch_size, window_size, des_size = data.shape
        all_descriptors = data
        new_descriptor = data
        h = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).type_as(data),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).type_as(data),
        )

        for i in range(step):
            # input_data = all_descriptors[:, i:, :]
            # print("input_data", input_data)
            with torch.no_grad():
                pred, h = self(new_descriptor, h)
            new_descriptor = pred[:, -1, :].reshape(batch_size, 1, des_size)
            # print("new_descriptor", new_descriptor)
            all_descriptors = torch.cat((all_descriptors, new_descriptor), 1)
        return all_descriptors.detach().cpu().numpy()[:, -step:, :]


# TODO
class LSTMEncoderDecoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters("config")
        descriptor_size = config.descriptor_size
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        dim_pos_encoding = config.dim_pos_encoding
        if dim_pos_encoding < 0:
            dim_pos_encoding = 0
        self.dim_pos_encoding = dim_pos_encoding

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            descriptor_size + dim_pos_encoding,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            1 + dim_pos_encoding, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, descriptor_size)

        self.pos_encoder = PositionalEncoding(
            dim_pos_encoding,
            dtype=self.linear.weight,
        )

        self.loss_function = nn.MSELoss()

        self.model_ep_loss_list = []

    def on_train_epoch_end(self, epoch_output=None):
        log_dict = {"epoch": self.current_epoch}

        loss = np.mean(self.model_ep_loss_list)
        log_dict["loss"] = loss

        wandb.log(log_dict, step=self.current_epoch)
        self.model_ep_loss_list = []

    def validation_epoch_end(self, validation_step_outputs):
        outputs = []
        for pred in validation_step_outputs:
            outputs.append(pred.detach().cpu().numpy())
        wandb.log({"val_loss": np.mean(outputs)}, step=self.current_epoch)

    def encode(self, x, pos_code=None):
        batch_size = x.shape[0]
        if pos_code is not None:
            x = torch.cat((x, pos_code), 2)
        h = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).type_as(x),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).type_as(x),
        )
        _, hidden = self.encoder(x, h)
        return hidden

    def decode(self, h, step, pos_code=None):
        batch_size = h[0].shape[1]
        x = torch.zeros(batch_size, step, 1).type_as(h[0])
        if pos_code is not None:
            x = torch.cat((x, pos_code), 2)
        out, h_new = self.decoder(x, h)
        out = out.reshape((batch_size * step, -1))
        out = self.linear(out)
        out = out.reshape((batch_size, step, -1))
        return out, h_new

    def forward(self, x, step):
        batch_size, seq_len, d_size = x.shape
        if self.dim_pos_encoding > 0:
            pos_code = self.pos_encoder.generate_encoding(
                batch_size, seq_len + step
            ).type_as(x)
            pos_code = torch.einsum("sbe->bse", pos_code)

        hidden = self.encode(x, pos_code[:, :seq_len])
        out, _ = self.decode(hidden, step, pos_code[:, seq_len:])
        return out

    def training_step(self, batch, batch_idx):
        data = batch[0]
        forecast_size = self.config.forecast_size
        data, target = data[:, :-forecast_size, :], data[:, -forecast_size:, :]
        pred = self(data, forecast_size)

        loss = self.loss_function(pred, target)

        self.model_ep_loss_list.append(loss.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        forecast_size = self.config.forecast_size
        data, target = data[:, :-forecast_size, :], data[:, -forecast_size:, :]
        pred = self(data, forecast_size)

        loss = self.loss_function(pred, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def predict(self, data, step):
        out = self(data, step)
        return out.detach().cpu().numpy()


class TransformerEncoderOnlyModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters("config")

        num_layers = config.num_layers
        descriptor_size = config.descriptor_size
        dim_pos_encoding = config.dim_pos_encoding
        nhead = config.nhead
        dropout = config.dropout
        dim_feedforward = config.dim_feedforward

        self.loss_function = nn.MSELoss()
        self.model = TransformerEncoderOnly(
            descriptor_size=descriptor_size,
            dim_pos_encoding=dim_pos_encoding,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )

        self.model_ep_loss_list = []

    def on_train_epoch_end(self, epoch_output=None):
        log_dict = {"epoch": self.current_epoch}

        loss = np.mean(self.model_ep_loss_list)
        log_dict["loss"] = loss

        wandb.log(log_dict, step=self.current_epoch)
        self.model_ep_loss_list = []

    def validation_epoch_end(self, validation_step_outputs):
        outputs = []
        for pred in validation_step_outputs:
            outputs.append(pred.detach().cpu().numpy())
        wandb.log({"val_loss": np.mean(outputs)}, step=self.current_epoch)

    def forward(self, src, src_mask):
        output = self.model(src, src_mask)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = self.model.generate_square_subsequent_mask(sz)
        return mask

    def training_step(self, batch, batch_idx):
        data = batch[0]
        data, target = data[:, :-1, :], data[:, 1:, :]
        # Note that data shape (BSE)
        src_mask = self.generate_square_subsequent_mask(data.size(1)).type_as(data)

        output = self.model(
            data,
            src_mask=src_mask,
        )

        loss = self.loss_function(output, target)

        self.model_ep_loss_list.append(loss.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        data, target = data[:, :-1, :], data[:, 1:, :]
        # Note that data shape (BSE)
        src_mask = self.generate_square_subsequent_mask(data.size(1)).type_as(data)

        output = self.model(
            data,
            src_mask=src_mask,
        )

        loss = self.loss_function(output, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

    def predict(self, data, step):
        all_descriptors = data
        batch_size, window_size, des_size = data.shape
        for i in range(step):
            input_data = all_descriptors
            # print("input_data", input_data)
            with torch.no_grad():
                src_mask = self.model.generate_square_subsequent_mask(
                    input_data.size(1)
                ).type_as(data)
                pred = self.model(input_data, src_mask=src_mask)
            new_descriptor = pred[:, -1, :].reshape(batch_size, 1, des_size)
            # print("new_descriptor", new_descriptor)
            all_descriptors = torch.cat((all_descriptors, new_descriptor), 1)
        return all_descriptors.detach().cpu().numpy()[:, -step:, :]


class TransformerModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters("config")
        descriptor_size = config.descriptor_size
        hidden_size = config.hidden_size
        num_layers = config.num_layers

        nhead = config.nhead
        dim_pos_encoding = config.dim_pos_encoding
        if dim_pos_encoding < 0:
            dim_pos_encoding = 0
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout

        d_model = descriptor_size + dim_pos_encoding

        self.dim_pos_encoding = dim_pos_encoding
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.pos_encoder = PositionalEncoding(
            dim_pos_encoding,
            dtype=self.transformer.decoder.layers[0].linear1.weight,
        )

        self.loss_function = nn.MSELoss()
        self.model_ep_loss_list = []

    def on_train_epoch_end(self, epoch_output=None):
        log_dict = {"epoch": self.current_epoch}

        loss = np.mean(self.model_ep_loss_list)
        log_dict["loss"] = loss

        wandb.log(log_dict, step=self.current_epoch)
        self.model_ep_loss_list = []

    def validation_epoch_end(self, validation_step_outputs):
        outputs = []
        for pred in validation_step_outputs:
            outputs.append(pred.detach().cpu().numpy())
        wandb.log({"val_loss": np.mean(outputs)}, step=self.current_epoch)

    def forward(self, x, step):
        batch_size, seq_len, d_size = x.shape
        x = torch.einsum("bse->sbe", x)
        tgt_input = torch.zeros((step, batch_size, d_size)).type_as(x)
        # the pos_code is in SBE
        if self.dim_pos_encoding > 0:
            pos_code = self.pos_encoder.generate_encoding(
                batch_size, seq_len + step
            ).type_as(x)
            x = torch.cat((x, pos_code[:seq_len]), 2)
            tgt_input = torch.cat((tgt_input, pos_code[seq_len:]), 2)

        src_mask = self.transformer.generate_square_subsequent_mask(seq_len).type_as(x)
        out = self.transformer(x, tgt_input, src_mask=src_mask)

        # remove pos_code in output
        out = out[:, :, :d_size]
        out = torch.einsum("sbe->bse", out)
        return out

    def training_step(self, batch, batch_idx):
        data = batch[0]
        forecast_size = self.config.forecast_size
        data, target = data[:, :-forecast_size, :], data[:, -forecast_size:, :]
        pred = self(data, forecast_size)

        loss = self.loss_function(pred, target)

        self.model_ep_loss_list.append(loss.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        forecast_size = self.config.forecast_size
        data, target = data[:, :-forecast_size, :], data[:, -forecast_size:, :]
        pred = self(data, forecast_size)

        loss = self.loss_function(pred, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def predict(self, data, step):
        out = self(data, step)
        return out.detach().cpu().numpy()


########################
#   components
########################


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        descriptor_size=5,
        dim_pos_encoding=250,
        nhead=10,
        num_encoder_layers=1,
        dropout=0.1,
        dim_feedforward=1024,
    ):
        super().__init__()

        self.dim_pos_encoding = dim_pos_encoding
        self.model_type = "Transformer"

        self.d_model = dim_pos_encoding + descriptor_size
        self.dropout = nn.Dropout2d(p=dropout)
        self.decoder = nn.Linear(self.d_model, descriptor_size)

        self.pos_encoder = PositionalEncoding(
            dim_pos_encoding,
            dtype=self.decoder.weight,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, src, src_mask):
        # batch/seq/E by default
        # src = torch.einsum("sbe->bse", src)
        src = self.dropout(src)
        src = torch.einsum("bse->sbe", src)

        # Adds a bit of noise during training, XXX not sure this is useful or not
        if self.training:
            src = src + torch.randn(src.shape).type_as(src) * 0.05

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # change order back to BSE
        output = torch.einsum("sbe->bse", output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask


# Note that this input order is "SBE"
class PositionalEncoding(nn.Module):
    def __init__(self, dim_pos_encoding, max_len=5000, dtype=torch.float64):
        super().__init__()
        pe = torch.zeros(max_len, dim_pos_encoding).type_as(dtype)
        position = torch.arange(0, max_len).type_as(dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_pos_encoding, 2)
            * (-torch.log(torch.tensor(10000.0)) / dim_pos_encoding)
        ).type_as(dtype)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        pe = self.pe[:seq_len, :].expand(-1, batch_size, -1)
        return torch.cat((x, pe), 2)

    def generate_encoding(self, batch_size, seq_len):
        pe = self.pe[:seq_len, :].expand(-1, batch_size, -1)
        return pe
