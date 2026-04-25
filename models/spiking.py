import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate
from torch.nn.init import xavier_uniform_, constant_


class SpikingMultiheadAttention(nn.Module):
    """
    Substituto neuromórfico para nn.MultiheadAttention.

    Cada projeção Q, K, V é seguida de um LIF independente.
    O potencial de membrana v_seq (análogo contínuo) é usado como
    representação para o scaled dot-product attention, preservando
    a dinâmica temporal dos spikes mas mantendo gradientes suaves.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            tau: float = 2.0,
            surrogate_fn=surrogate.Sigmoid(alpha=4.0),
            dropout: float = 0.0,
            bias: bool = True,
            batch_first: bool = True,
            use_mem_seq: bool = True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim deve ser divisível por num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.use_mem_seq = use_mem_seq
        self.scale = math.sqrt(self.head_dim)

        factory = {"device": device, "dtype": dtype}

        # Projeções lineares separadas
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)

        # Um LIF por stream Q/K/V — parâmetros tau independentes
        lif_kwargs = dict(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='m',
            store_v_seq=True,
        )
        self.lif_q = neuron.LIFNode(**lif_kwargs)
        self.lif_k = neuron.LIFNode(**lif_kwargs)
        self.lif_v = neuron.LIFNode(**lif_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.out_proj.weight)
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if proj.bias is not None:
                constant_(proj.bias, 0.0)

    def _reset_lif_states(self):
        """Reseta os estados dos neurônios entre batches."""
        self.lif_q.reset()
        self.lif_k.reset()
        self.lif_v.reset()

    def _project_and_spike(
            self,
            x: torch.Tensor,
            proj: nn.Linear,
            lif: neuron.LIFNode,
    ) -> torch.Tensor:
        """Aplica projeção linear + LIF em modo multi-step."""
        cur = proj(x)
        cur_t = cur.permute(1, 0, 2)
        spk_t = lif(cur_t)

        if self.use_mem_seq:
            out_t = lif.v_seq
        else:
            out_t = spk_t

        return out_t.permute(1, 0, 2)

    def _prepare_attn_mask(self, attn_mask, key_padding_mask, B, T_q, T_k):
        """Combina máscaras em matrizes aditivas compatíveis com o formato do PyTorch"""
        merged = None

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [T_q, T_k] -> [1, 1, T_q, T_k]
                merged = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B*h, T_q, T_k] -> [B, h, T_q, T_k]
                merged = attn_mask.view(B, self.num_heads, T_q, T_k)

            # Garante que seja máscara aditiva (-inf para ignorar)
            if merged.dtype == torch.bool:
                merged = torch.zeros_like(merged, dtype=torch.float).masked_fill_(merged, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask tem formato [B, T_k]
            # Convertido para [B, 1, 1, T_k] para usar broadcasting nativo
            kpm = key_padding_mask.view(B, 1, 1, T_k)

            # Converte booleano para aditivo
            if kpm.dtype == torch.bool:
                kpm_additive = torch.zeros_like(kpm, dtype=torch.float).masked_fill_(kpm, float('-inf'))
            else:
                kpm_additive = torch.zeros_like(kpm, dtype=torch.float).masked_fill_(kpm > 0, float('-inf'))

            merged = kpm_additive if merged is None else merged + kpm_additive

        return merged

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask=None,
            need_weights: bool = True,
            attn_mask=None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        self._reset_lif_states()

        # Projeções neuromórficas independentes
        Q = self._project_and_spike(query, self.q_proj, self.lif_q)
        K = self._project_and_spike(key, self.k_proj, self.lif_k)
        V = self._project_and_spike(value, self.v_proj, self.lif_v)

        def reshape_heads(x, seq_len):
            return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = reshape_heads(Q, T_q)  # [B, h, T_q, d_k]
        K = reshape_heads(K, T_k)  # [B, h, T_k, d_k]
        V = reshape_heads(V, T_k)  # [B, h, T_k, d_k]

        dropout_p = self.dropout if self.training else 0.0
        attn_mask_merged = self._prepare_attn_mask(attn_mask, key_padding_mask, B, T_q, T_k)

        # SEPARAÇÃO DE ROTAS (Escala e pesos corretos)
        if not need_weights:
            # Rota ultra-rápida (FlashAttention / Memory Efficient nativa)
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask_merged,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            attn_weights = None
        else:
            # Rota manual para exportar os pesos (Math Attention)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # Aplicação correta e única da escala

            if is_causal:
                causal_mask = torch.triu(torch.full((T_q, T_k), float('-inf'), device=Q.device), diagonal=1)
                scores = scores + causal_mask.view(1, 1, T_q, T_k)

            if attn_mask_merged is not None:
                scores = scores + attn_mask_merged

            attn_weights_raw = torch.softmax(scores, dim=-1)
            attn_weights_drop = F.dropout(attn_weights_raw, p=dropout_p, training=self.training)

            attn_output = torch.matmul(attn_weights_drop, V)

            if average_attn_weights:
                attn_weights = attn_weights_raw.mean(dim=1)
            else:
                attn_weights = attn_weights_raw

        # Reagrupa as cabeças
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights


class SJ_Spiking_MultiStep_Attention(nn.Module):
    """
    ARQUITETURA CAMPEÃ: SNN + Analog Attention no Tempo
    """

    def __init__(
            self,
            time_steps=50,
            n_features=3,
            d_model=16,
            num_heads=4,
            tau=3.0,
            surrogate_fn=surrogate.Sigmoid(alpha=2.0)
    ):
        super().__init__()
        self.time_steps = time_steps
        self.feature_proj = nn.Linear(n_features, d_model)

        self.lif_encoder = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='m',
            store_v_seq=True,
        )

        self.attention = SpikingMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            tau=tau,
            surrogate_fn=surrogate_fn,
            dropout=0.0,
            batch_first=True,
            use_mem_seq=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        cur = self.feature_proj(x)
        cur_t = cur.permute(1, 0, 2)

        self.lif_encoder.reset()
        _ = self.lif_encoder(cur_t)
        mem_sequence = self.lif_encoder.v_seq.permute(1, 0, 2)

        attn_out, _ = self.attention(mem_sequence, mem_sequence, mem_sequence)

        x_time = self.norm(mem_sequence + attn_out)
        pooled = torch.mean(x_time, dim=1)
        return self.fc_out(pooled)


class SJ_Spiking_Attention(nn.Module):
    """
    Atenção nas Features (Tokens = features físicas: V, I, T, SoC).
    """

    def __init__(
            self,
            time_steps=50,
            n_features=3,
            d_model=64,
            num_heads=4,
            tau=2.0,
            surrogate_fn=surrogate.Sigmoid(alpha=4.0)
    ):
        super().__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        self.d_model = d_model

        self.feature_proj = nn.Linear(1, d_model)

        self.lif = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='m',
            store_v_seq=False,
        )

        self.attention = SpikingMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            tau=tau,
            surrogate_fn=surrogate_fn,
            batch_first=True,
            use_mem_seq=True,
        )
        self.norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        B, T, F = x.shape

        x_expanded = x.unsqueeze(-1)
        cur = self.feature_proj(x_expanded)

        cur_t = cur.permute(1, 0, 2, 3).reshape(T, B * F, self.d_model)

        self.lif.reset()
        self.lif(cur_t)

        mem_final = self.lif.v.reshape(B, F, self.d_model)

        attn_out, _ = self.attention(mem_final, mem_final, mem_final)
        x_feat = self.norm(mem_final + attn_out)

        pooled = x_feat.mean(dim=1)
        return self.fc_out(pooled)


class SJ_Spiking_Hybrid(nn.Module):
    """
    Hybrid SNN-Transformer (Atenção Simultânea no Tempo e nas Features)
    """

    def __init__(self,
                 time_steps: int,
                 n_features: int,
                 d_model: int = 64,
                 num_heads: int = 4,
                 fusion_hidden: int = 64,
                 tau: float = 2.0,
                 surrogate_fn=surrogate.Sigmoid(alpha=4.0)
                 ):
        super().__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        self.d_model = d_model

        self.proj_feat = nn.Linear(1, d_model)
        self.proj_time = nn.Linear(n_features, d_model)

        self.lif_feat = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='m',
            store_v_seq=False,
        )
        self.lif_time = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='m',
            store_v_seq=True,
        )

        self.attn_feat = SpikingMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            tau=tau,
            surrogate_fn=surrogate_fn,
            batch_first=True,
            use_mem_seq=True,
        )
        self.norm_feat = nn.LayerNorm(d_model)

        self.attn_time = SpikingMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            tau=tau,
            surrogate_fn=surrogate_fn,
            batch_first=True,
            use_mem_seq=True,
        )
        self.norm_time = nn.LayerNorm(d_model)

        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape

        # RAMO A — Features
        x_a = x.unsqueeze(-1)
        cur_a = self.proj_feat(x_a)
        cur_a_t = cur_a.permute(1, 0, 2, 3).reshape(T, B * F, self.d_model)

        self.lif_feat.reset()
        self.lif_feat(cur_a_t)

        mem_feat = self.lif_feat.v.reshape(B, F, self.d_model)

        af_out, _ = self.attn_feat(mem_feat, mem_feat, mem_feat)
        af_out = self.norm_feat(mem_feat + af_out)
        pool_feat = af_out.mean(dim=1)

        # RAMO B — Tempo
        cur_b = self.proj_time(x)
        cur_b_t = cur_b.permute(1, 0, 2)

        self.lif_time.reset()
        self.lif_time(cur_b_t)

        mem_seq = self.lif_time.v_seq.permute(1, 0, 2)

        at_out, _ = self.attn_time(mem_seq, mem_seq, mem_seq)
        at_out = self.norm_time(mem_seq + at_out)
        pool_time = at_out.mean(dim=1)

        # FUSÃO
        fused = torch.cat([pool_feat, pool_time], dim=1)
        return self.fusion(fused)


class SimpleSNN(nn.Module):
    """
    Arquitetura SNN feedforward simples para processamento de séries temporais.
    Extrai características temporais através da dinâmica de vazamento (LIF)
    sem a complexidade de mecanismos de atenção.
    """

    def __init__(
            self,
            time_steps: int = 50,
            n_features: int = 3,
            hidden_dim: int = 64,
            tau: float = 2.0,
            surrogate_fn=surrogate.Sigmoid(alpha=4.0)
    ):
        super().__init__()
        self.time_steps = time_steps

        # Projeção inicial das características físicas
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='m'
        )

        # Segunda camada oculta
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='m',
            store_v_seq=True  # Armazenamos a sequência de membrana final para o pooling
        )

        # Head de regressão final
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Time, Features]

        # SpikingJelly em step_mode='m' espera o tempo na dimensão 0: [Time, Batch, Features]
        x_t = x.permute(1, 0, 2)

        # Resetar os estados de membrana dos neurônios a cada forward
        self.lif1.reset()
        self.lif2.reset()

        # Passagem pela primeira camada
        out_t = self.fc1(x_t)
        out_t = self.lif1(out_t)  # Saída aqui são spikes binários

        # Passagem pela segunda camada
        out_t = self.fc2(out_t)
        out_t = self.lif2(out_t)

        # Para regressão, usar a média do potencial de membrana ao longo do tempo
        # geralmente gera gradientes mais estáveis do que fazer pooling direto nos spikes.
        mem_seq = self.lif2.v_seq  # shape: [Time, Batch, hidden_dim]

        # Voltando para [Batch, Time, hidden_dim] e tirando a média no eixo do tempo
        mem_seq = mem_seq.permute(1, 0, 2)
        pooled = mem_seq[:, -1, :]

        # Projeção final
        return self.fc_out(pooled)


# ---------------------------------------------------------------------------
# LSM — Liquid State Machine (Reservatório Esparso + Readout Linear)
# ---------------------------------------------------------------------------
class SJ_LiquidStateMachine(nn.Module):
    """
    Liquid State Machine neuromórfica com reservatório recorrente esparso.

    O reservatório (liquid) é uma rede recorrente de neurônios LIF com pesos
    aleatórios FIXOS (não treináveis). Apenas o readout linear é treinado.
    A riqueza dinâmica do liquid transforma a série temporal em uma
    representação de alta dimensão, da qual o readout extrai o SoH.

    Parâmetros
    ----------
    reservoir_size : int
        Número de neurônios no liquid. Maior = mais expressivo, mais lento.
    sparsity : float
        Fração de conexões ZERADAS no reservatório (0.8 = 80% esparso).
    spectral_radius : float
        Escala os pesos recorrentes para controlar o "eco" do reservatório.
        Valores < 1.0 garantem estabilidade (Echo State Property).
    input_scaling : float
        Ganho aplicado aos pesos de entrada para o liquid.
    """

    def __init__(
            self,
            time_steps: int = 50,
            n_features: int = 3,
            reservoir_size: int = 256,
            sparsity: float = 0.8,
            spectral_radius: float = 0.9,
            input_scaling: float = 0.5,
            tau: float = 2.0,
            surrogate_fn=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.time_steps = time_steps
        self.reservoir_size = reservoir_size

        # --- Projeção de entrada (treinável) ---
        self.input_proj = nn.Linear(n_features, reservoir_size, bias=False)

        # --- Reservatório recorrente (FIXO — não treinável) ---
        W_res = torch.randn(reservoir_size, reservoir_size)

        # Aplica esparsidade: zera uma fração das conexões
        mask = torch.rand_like(W_res) > sparsity
        W_res = W_res * mask.float()

        # Normaliza pelo raio espectral para garantir Echo State Property
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        W_res = W_res / (eigenvalues.max() + 1e-8) * spectral_radius

        # Registra como buffer: salvo no state_dict, mas sem gradiente
        self.register_buffer("W_res", W_res)

        # Escala dos pesos de entrada
        self.input_scaling = input_scaling

        # --- Neurônio LIF do reservatório ---
        self.lif_reservoir = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate_fn,
            step_mode='s',  # step-by-step: controlamos o loop manualmente
            store_v_seq=False,
        )

        # --- Readout treinável ---
        self.readout = nn.Sequential(
            nn.LayerNorm(reservoir_size),
            nn.Linear(reservoir_size, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, _ = x.shape

        self.lif_reservoir.reset()

        # Estado de spike acumulado para readout por mean-pooling temporal
        spike_acc = torch.zeros(B, self.reservoir_size, device=x.device)

        for t in range(T):
            x_t = x[:, t, :]  # [B, F]

            # Corrente de entrada: projeção linear escalada
            i_input = self.input_scaling * self.input_proj(x_t)  # [B, reservoir_size]

            # Corrente recorrente: spike anterior * W_res
            i_rec = spike_acc @ self.W_res.T  # [B, reservoir_size]

            # Integra as duas correntes no LIF
            spk = self.lif_reservoir(i_input + i_rec)  # [B, reservoir_size]

            spike_acc = spk

        # Usa o potencial de membrana final como representação do liquid
        # (mais suave para regressão do que os spikes binários)
        liquid_state = self.lif_reservoir.v  # [B, reservoir_size]

        return self.readout(liquid_state)


# ---------------------------------------------------------------------------
# 5. SNN Dilated — WaveNet-style Neuromórfico (Padrões locais multi-escala)
# ---------------------------------------------------------------------------
class SJ_Spiking_Dilated(nn.Module):
    """
    Versão neuromórfica da DilatedCNN.

    Cada bloco dilatado substitui o GELU por um neurônio LIF independente,
    preservando a dinâmica de dilatação exponencial (1, 2, 4, 8...) para
    captura multiescala, mas agora com codificação esparsa por spikes.

    O potencial de membrana v_seq (análogo contínuo) é usado nas conexões
    residuais em vez dos spikes binários — mesma estratégia do
    SJ_Spiking_MultiStep_Attention — para manter gradientes suaves.
    """

    def __init__(
            self,
            time_steps: int,
            n_features: int,
            d_model: int = 64,
            n_layers: int = 4,
            tau: float = 2.0,
            surrogate_fn=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.n_layers = n_layers

        # Projeção inicial: [B, F, T] -> [B, d_model, T]
        # Conv1d padrão, sem LIF — apenas alinha o espaço de features
        self.input_proj = nn.Conv1d(n_features, d_model, kernel_size=1)

        # Convoluções dilatadas (sem ativação — o LIF fará esse papel)
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=2 ** i,
                    dilation=2 ** i,
                ),
                nn.BatchNorm1d(d_model),
            )
            for i in range(n_layers)
        ])

        # Um LIF por camada dilatada — parâmetros tau independentes
        # step_mode='m': recebe [T, B*d_model] e processa todo o tempo de uma vez
        self.lif_layers = nn.ModuleList([
            neuron.LIFNode(
                tau=tau,
                surrogate_function=surrogate_fn,
                step_mode='m',
                store_v_seq=True,  # v_seq usado na conexão residual
            )
            for _ in range(n_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _reset_lif_states(self):
        for lif in self.lif_layers:
            lif.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, _ = x.shape

        # [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)

        # Projeção inicial (sem spike)
        x = self.input_proj(x)  # [B, d_model, T]

        self._reset_lif_states()

        for conv_block, lif in zip(self.dilated_convs, self.lif_layers):
            # 1. Convolução dilatada + BN: [B, d_model, T]
            conv_out = conv_block(x)

            # 2. Prepara para o LIF: Conv1d usa [B, C, T], LIF multi-step usa [T, B*C]
            #    Reshape: [B, d_model, T] -> [T, B, d_model]
            conv_t = conv_out.permute(2, 0, 1)  # [T, B, d_model]

            # 3. Passa pelo LIF — gera spikes e acumula v_seq
            lif(conv_t)  # [T, B, d_model]

            # 4. Usa o potencial de membrana (análogo contínuo) para o residual
            #    v_seq shape: [T, B, d_model] -> [B, d_model, T]
            mem_seq = lif.v_seq.permute(1, 2, 0)  # [B, d_model, T]

            # 5. Conexão residual: preserva informação de escalas anteriores
            x = x + mem_seq

        # Pooling global: [B, d_model, T] -> [B, d_model]
        x = self.global_pool(x).squeeze(-1)

        return self.regressor(x)
