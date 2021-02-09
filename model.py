import math

import torch
import torch.nn as nn

NUM_FEAT = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim_pos_encoding = 50
nhead = 5  # the number of heads in the multiheadattention models
dropout = 0.1
positional_encoding_dropout = 0.0
num_encoder_layers = 1
num_decoder_layers = 1
dim_feedforward = 128
seq_len = 40


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        dim_pos_encoding=250,
        nhead=10,
        num_encoder_layers=1,
        dropout=0.1,
        dim_feedforward=1024,
    ):
        super().__init__()
        self.dim_pos_encoding = dim_pos_encoding
        self.model_type = "Transformer"

        self.d_model = dim_pos_encoding + NUM_FEAT
        self.dropout = nn.Dropout2d(p=dropout)
        self.decoder = nn.Linear(self.d_model, NUM_FEAT)

        self.pos_encoder = PositionalEncoding(
            dim_pos_encoding, dropout=positional_encoding_dropout
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

        src = torch.einsum("sbe->bse", src)
        src = self.dropout(src)
        src = torch.einsum("bse->sbe", src)

        # Adds a bit of noise during training, XXX not sure this is useful or not
        if self.training:
            src = src + torch.randn(src.shape).to(device) * 0.05

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, dim_pos_encoding, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim_pos_encoding)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_pos_encoding, 2).float()
            * (-math.log(10000.0) / dim_pos_encoding)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        pe = self.pe[:seq_len, :].expand(-1, batch_size, -1)
        pe = self.dropout(pe)
        return torch.cat((x, pe), 2)


model = TransformerEncoderOnly(
    dim_pos_encoding=dim_pos_encoding,
    nhead=nhead,
    dropout=dropout,
    num_encoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
).to(device)
