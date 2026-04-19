import torch
import torch.nn as nn


class PyTorch_iTransformer(nn.Module):
    def __init__(self, time_steps, n_features, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.feature_proj = nn.Linear(time_steps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(n_features * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_inv = x.permute(0, 2, 1)
        tokens = self.feature_proj(x_inv)
        enc_out = self.transformer(tokens)
        return self.fc_out(enc_out.flatten(1))


class PyTorch_DynamicGraph_iTR(nn.Module):
    def __init__(self, time_steps, n_features, d_model=64):
        super().__init__()
        self.itransformer = PyTorch_iTransformer(time_steps, n_features, d_model)
        self.gnn_conv = nn.Conv1d(n_features, d_model, kernel_size=3, padding=1)
        self.gnn_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Sequential(
            nn.Linear(d_model + (n_features * d_model), 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, x):
        z = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(z, z.transpose(1, 2))
        var = torch.diagonal(cov, dim1=1, dim2=2) + 1e-6
        std = torch.sqrt(var.unsqueeze(2) * var.unsqueeze(1))
        adj = cov / std

        agg = torch.bmm(adj, x) + x
        agg_inv = agg.permute(0, 2, 1)
        gnn_feat = torch.relu(self.gnn_conv(agg_inv))
        gnn_pool = self.gnn_pool(gnn_feat).squeeze(-1)

        x_inv = x.permute(0, 2, 1)
        tokens = self.itransformer.feature_proj(x_inv)
        enc_out = self.itransformer.transformer(tokens).flatten(1)

        fused = torch.cat([gnn_pool, enc_out], dim=1)
        return self.fusion(fused)
