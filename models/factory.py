from .classic_dl import PyTorch_CNN1D, PyTorch_LSTM, PyTorch_CNN_LSTM
from .physics import PyTorch_Phys_iTransformer, PyTorch_Phys_iTransformer_Curriculum
from .spiking import SJ_Spiking_MultiStep_Attention, SJ_Spiking_Attention, \
    SJ_Spiking_Hybrid
from .transformers import PyTorch_iTransformer, PyTorch_DynamicGraph_iTR
from spikingjelly.activation_based import surrogate


def get_surrogate_fn(config):
    """Auxiliar para converter a string de configuração na função surrogate real."""
    alpha = config.surrogate_alpha
    name = config.surrogate

    if name == 'ATan':
        return surrogate.ATan(alpha=alpha)
    elif name == 'SoftSign':
        return surrogate.SoftSign(alpha=alpha)
    elif name == 'Sigmoid':
        return surrogate.Sigmoid(alpha=alpha)
    return surrogate.ATan(alpha=3.5)  # Default de segurança


def get_model_factory(config, n_features):
    """
    Instancia todos os modelos definidos para experimentação usando os
    hiperparâmetros otimizados do PipelineConfig.
    """
    surr_fn = get_surrogate_fn(config)

    return {
        # Modelos Neuromórficos (SNNs) - Agora totalmente dinâmicos
        "SJ-Spiking-MultiStep": SJ_Spiking_MultiStep_Attention(
            time_steps=config.time_steps,
            n_features=n_features,
            d_model=config.d_model,
            num_heads=config.num_heads,
            tau=config.tau,
            surrogate_fn=surr_fn
        ).to(config.device),

        "SJ-Spiking-Attention": SJ_Spiking_Attention(
            time_steps=config.time_steps,
            n_features=n_features,
            d_model=config.d_model,
            num_heads=config.num_heads,
            tau=config.tau
        ).to(config.device),

        "SJ-Spiking-Hybrid": SJ_Spiking_Hybrid(
            time_steps=config.time_steps,
            n_features=n_features,
            d_model=config.d_model,
            num_heads=config.num_heads,
            tau=config.tau
        ).to(config.device),

        # Modelos de Deep Learning Clássico e Transformers
        "CNN-1D": PyTorch_CNN1D(config.time_steps, n_features).to(config.device),
        "LSTM": PyTorch_LSTM(config.time_steps, n_features).to(config.device),
        "CNN-LSTM": PyTorch_CNN_LSTM(config.time_steps, n_features).to(config.device),

        "iTransformer": PyTorch_iTransformer(
            config.time_steps,
            n_features,
            d_model=config.d_model
        ).to(config.device),

        "DynamicGraph-iTR": PyTorch_DynamicGraph_iTR(
            config.time_steps,
            n_features,
            d_model=config.d_model
        ).to(config.device),

        "Phys-iTR-Curriculum": PyTorch_Phys_iTransformer_Curriculum(
            config.time_steps,
            n_features,
            d_model=config.d_model
        ).to(config.device),

        "Phys-iTransformer": PyTorch_Phys_iTransformer(
            config.time_steps,
            n_features,
            d_model=config.d_model
        ).to(config.device),
    }