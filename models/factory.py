from .classic_dl import PyTorch_CNN1D, PyTorch_LSTM, PyTorch_CNN_LSTM
from .physics import PyTorch_Phys_iTransformer, PyTorch_Phys_iTransformer_Curriculum
from .spiking import SJ_Spiking_MultiStep_Attention, SJ_Spiking_Attention, \
    SJ_Spiking_Hybrid
from .transformers import PyTorch_iTransformer, PyTorch_DynamicGraph_iTR
from spikingjelly.activation_based import surrogate

def get_model_factory(config, n_features):
    """Instancia todos os modelos definidos para experimentação."""
    return {
        "SJ-Spiking-MultiStep": SJ_Spiking_MultiStep_Attention(config.time_steps, n_features, surrogate_fn=surrogate.ATan(alpha=3.5)).to(config.device),
        "SJ-Spiking-Attention": SJ_Spiking_Attention(config.time_steps, n_features).to(config.device),
        "SJ-Spiking-Hybrid": SJ_Spiking_Hybrid(config.time_steps, n_features).to(config.device),
        "CNN-1D": PyTorch_CNN1D(config.time_steps, n_features).to(config.device),
        "LSTM": PyTorch_LSTM(config.time_steps, n_features).to(config.device),
        "CNN-LSTM": PyTorch_CNN_LSTM(config.time_steps, n_features).to(config.device),
        "iTransformer": PyTorch_iTransformer(config.time_steps, n_features).to(config.device),
        "DynamicGraph-iTR": PyTorch_DynamicGraph_iTR(config.time_steps, n_features).to(config.device),
        "Phys-iTR-Curriculum": PyTorch_Phys_iTransformer_Curriculum(config.time_steps, n_features).to(config.device),
        "Phys-iTransformer": PyTorch_Phys_iTransformer(config.time_steps, n_features).to(config.device),
    }
