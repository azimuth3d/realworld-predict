import torch
import torch.nn as nn
import numpy as np

class MultiStepTemporalTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1, forecast_horizon=12):
        super(MultiStepTemporalTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_horizon)
        )

    def forward(self, x):
        seq_len = x.size(1)
        x_proj = self.input_projection(x) + self.pos_encoder[:, :seq_len, :]
        out = self.transformer_encoder(x_proj)
        out = self.fc_out(out[:, -1, :])
        return out
