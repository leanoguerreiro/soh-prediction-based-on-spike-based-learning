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
            surrogate_alpha=4.0,
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
                 surrogate_alpha: float = 4.0,
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
            surrogate_function=surrogate.Sigmoid(alpha=surrogate_alpha),
            step_mode='m',
            store_v_seq=False,
        )
        self.lif_time = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate.Sigmoid(alpha=surrogate_alpha),
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