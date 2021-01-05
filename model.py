import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

NUM_FEAT = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 50
num_heads = 10  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
num_encoder_layers = 1
num_decoder_layers = 1
dim_feedforward = 512

bptt = 200


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=250,
        num_heads=10,
        num_encoder_layers=1,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        self.model_type = "Transformer"

        self.encoder = nn.Linear(NUM_FEAT, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        self.decoder = nn.Linear(d_model, NUM_FEAT)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):

        if self.training:
            src = src + torch.randn(src.shape).to(device) * 0.1

        src = self.encoder(src) * math.sqrt(NUM_FEAT)

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


class MyTransfomer(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Linear(NUM_FEAT, d_model)
        self.outbedding = nn.Linear(d_model, NUM_FEAT)

        self.transformer = nn.Transformer(d_model=d_model, **kwargs)

    def generate_square_subsequent_mask(self, sz):
        return self.transformer.generate_square_subsequent_mask(sz)

    def generate_no_peak_mask(self, sz):
        # mask = torch.from_numpy(np.triu(np.ones(sz)).astype(np.bool))
        # return mask
        # nopeak_mask = np.triu(np.ones([sz, sz]), k=1).astype("uint8")
        # return torch.from_numpy(nopeak_mask) == 0
        mask = torch.triu(torch.ones(sz, sz), diagonal=0).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
        # return self.generate_square_subsequent_mask(sz)

    def forward(self, src, target, src_mask, tgt_mask):

        src = self.embedding(src)
        target = self.embedding(target)

        out = self.transformer(src, target, src_mask=src_mask, tgt_mask=tgt_mask)

        out = self.outbedding(out)

        return out


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward,
        num_encoder_layers,
        dropout=0.5,
    ):
        super().__init__()

        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers
        )
        self.encoder = nn.Linear(NUM_FEAT, d_model)
        # self.ninp = ninp
        # self.decoder = nn.Linear(ninp, NUM_FEAT)

        # decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        # self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        #
        self.decoder = nn.Linear(d_model, NUM_FEAT)

        # self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src,
        src_mask,
    ):

        if self.training:
            src = src + torch.randn(src.shape).to(device) * 0.1

        src = self.encoder(src) * math.sqrt(NUM_FEAT)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # output = self.transformer_decoder(output, tgt_mask)
        output = self.decoder(output)
        return output

    # def forward(self, src, src_mask):
    #     return src


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print("x", x.shape, "pe", self.pe.shape)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


model = TransformerModel(
    d_model=d_model,
    num_heads=num_heads,
    dim_feedforward=dim_feedforward,
    num_encoder_layers=num_encoder_layers,
    dropout=dropout,
).to(device)

# model = nn.Transformer(d_model=NUM_FEAT, nhead=5).to(device)
# model = MyTransfomer(
#     d_model=d_model,
#     nhead=nhead,
#     dropout=dropout,
#     num_decoder_layers=num_decoder_layers,
#     num_encoder_layers=num_decoder_layers,
#     dim_feedforward=dim_feedforward,
# ).to(device)

# model = TransformerEncoderOnly(
#     d_model=d_model,
#     num_heads=num_heads,
#     dim_feedforward=dim_feedforward,
#     num_encoder_layers=num_encoder_layers,
#     dropout=dropout,
# ).to(device)

print(model)
