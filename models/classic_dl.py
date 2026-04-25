import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. CNN 1D (Foco em extração de padrões locais de alta frequência)
# ---------------------------------------------------------------------------
class PyTorch_CNN1D(nn.Module):
    def __init__(self, time_steps, n_features, d_model=64):
        super().__init__()
        # Conv1d espera entrada: [Batch, Channels (Features), Time]
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model * 2),
            nn.ReLU(),
        )

        # AdaptiveAvgPool garante que a saída seja sempre do mesmo tamanho,
        # independente do tamanho da janela de tempo após o pooling.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Transforma de [B, T, F] para [B, F, T]
        x_permuted = x.permute(0, 2, 1)

        # Extração de features: [B, F, T] -> [B, d_model*2, T']
        features = self.feature_extractor(x_permuted)

        # Pooling global: [B, d_model*2, T'] -> [B, d_model*2, 1] -> [B, d_model*2]
        pooled = self.global_pool(features).squeeze(-1)

        # Regressão do SoH
        return self.regressor(pooled)


# ---------------------------------------------------------------------------
# 2. LSTM (Foco em dependências temporais de longo prazo)
# ---------------------------------------------------------------------------
class PyTorch_LSTM(nn.Module):
    def __init__(self, time_steps, n_features, hidden_size=64, num_layers=2):
        super().__init__()
        # batch_first=True mantém a entrada no nosso padrão natural [B, T, F]
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # out: contém todos os hidden states do último layer [B, T, hidden_size]
        # hn: hidden state final
        # cn: cell state final
        out, (hn, cn) = self.lstm(x)

        # Queremos a predição baseada no último instante de tempo observado
        last_time_step_out = out[:, -1, :]  # [B, hidden_size]

        return self.regressor(last_time_step_out)


# ---------------------------------------------------------------------------
# 3. CNN-LSTM Híbrida (O melhor dos dois mundos)
# ---------------------------------------------------------------------------
class PyTorch_CNN_LSTM(nn.Module):
    def __init__(self, time_steps, n_features, cnn_out=32, lstm_hidden=64):
        super().__init__()

        # Extrator local (CNN)
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, cnn_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Modelador sequencial (LSTM)
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 1. CNN processa a dimensão das features ao longo do tempo
        x_permuted = x.permute(0, 2, 1)  # [B, F, T]
        cnn_features = self.conv_block(x_permuted)  # [B, cnn_out, T/2]

        # 2. Re-preparamos para a LSTM: [B, T/2, cnn_out]
        # Agora as features extraídas pela CNN agem como a nova entrada da LSTM
        lstm_input = cnn_features.permute(0, 2, 1)

        # 3. LSTM processa a sequência temporal reduzida
        out, _ = self.lstm(lstm_input)
        last_out = out[:, -1, :]

        # 4. Predição
        return self.regressor(last_out)


# ---------------------------------------------------------------------------
# 4. Dilated Causal CNN — WaveNet-style (Padrões locais multi-escala)
# ---------------------------------------------------------------------------
class PyTorch_DilatedCNN(nn.Module):
    """
    Rede convolucional com dilatação exponencial.
    Cada camada dobra o dilation_rate, permitindo que o campo receptivo
    cresça exponencialmente (1, 2, 4, 8...) sem aumentar parâmetros.
    Ideal para capturar padrões locais em múltiplas escalas de frequência.
    """

    def __init__(self, time_steps, n_features, d_model=64, n_layers=4):
        super().__init__()

        # Projeção inicial das features para o espaço d_model
        self.input_proj = nn.Conv1d(n_features, d_model, kernel_size=1)

        # Stack de convoluções dilatadas com dilation = 2^i
        # Cada bloco tem sua própria dilatação, BatchNorm e conexão residual
        self.dilated_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=2 ** i,  # padding = dilation mantém o tamanho temporal
                    dilation=2 ** i,  # campo receptivo dobra a cada camada
                ),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )
            for i in range(n_layers)  # dilations: 1, 2, 4, 8
        ])

        # Fusão multiescala: agrega informação de todas as camadas
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)

        # Projeção inicial
        x = self.input_proj(x)  # [B, d_model, T]

        # Passa por cada camada dilatada com conexão residual
        for layer in self.dilated_layers:
            x = x + layer(x)  # residual: preserva informação de escalas anteriores

        # Pooling global e regressão
        x = self.global_pool(x).squeeze(-1)  # [B, d_model]
        return self.regressor(x)
