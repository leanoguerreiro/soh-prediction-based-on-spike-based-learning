import logging
import os
import random

import numpy as np
import torch
import torch.ao.quantization


def setup_logger(name="BatteryPipeline"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Opcional: Salvar em arquivo
        # fh = logging.FileHandler('pipeline.log')
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)
    return logger


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def add_gaussian_noise(tensor, std):
    if std == 0:
        return tensor
    noise = torch.randn_like(tensor) * std

    noise = torch.clamp(noise, min=-3 * std, max=3 * std)

    return tensor + noise


def apply_post_training_quantization(trained_model, dummy_input):
    """PTQ para deployment em BMS (Edge AI)."""
    import torch.nn as nn

    logger = logging.getLogger("BatteryPipeline")
    logger.info("Aplicando Post-Training Quantization (PTQ)...")

    model_cpu = trained_model.to('cpu').eval()
    dummy_input_cpu = dummy_input.to('cpu')

    def get_model_size_mb(model):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size

    size_fp32 = get_model_size_mb(model_cpu)

    # Quantização Dinâmica nas camadas lineares
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )

    size_int8 = get_model_size_mb(quantized_model)
    logger.info(f"Modelo Float32: {size_fp32:.4f} MB | Modelo Int8: {size_int8:.4f} MB")
    logger.info(f"Redução de tamanho: {(1 - size_int8 / size_fp32) * 100:.1f}%")

    return quantized_model
