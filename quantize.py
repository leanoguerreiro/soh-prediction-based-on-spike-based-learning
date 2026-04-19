import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from spikingjelly.activation_based import functional

# Importações do seu ecossistema modular
from config.settings import PipelineConfig
from utils.common import setup_logger, set_seed, apply_post_training_quantization
from data.ingestion import process_nasa_dataset
from features.builder import build_sequences
from models.factory import get_model_factory


def get_model_size_mb(model):
    """Função auxiliar para obter o tamanho do modelo em MB para o relatório."""
    torch.save(model.state_dict(), "temp_size.p")
    size = os.path.getsize("temp_size.p") / 1e6
    os.remove("temp_size.p")
    return size


def get_holdout_test_data(config):
    """Recria a divisão Holdout exata usando a lógica do Cofre do main.py."""
    df = process_nasa_dataset(config)
    X, y, groups = build_sequences(df, config)
    n_features = X.shape[2]

    # 1. Separação Global (Dev vs Cofre)
    gss_global = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    dev_idx, holdout_idx = next(gss_global.split(X, y, groups))

    X_dev, y_dev, groups_dev = X[dev_idx], y[dev_idx], groups[dev_idx]
    X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

    # 2. Fit do Scaler apenas no Treino do Dev (exatamente como feito no holdout.py)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.seed)
    train_idx, val_idx = next(gss_val.split(X_dev, y_dev, groups_dev))
    X_train = X_dev[train_idx]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    X_test_s = scaler.transform(X_holdout.reshape(-1, n_features)).reshape(X_holdout.shape)

    # Converte para Tensor (CPU, pois quantização dinâmica foca em Edge/Microcontroladores)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to('cpu')
    return X_test_t, y_holdout, n_features


def evaluate_inference(model, X_test_t, y_test):
    """Avalia o tempo de inferência e as métricas do modelo na CPU."""
    model.eval()

    # Aquecimento (Warm-up) para medição de tempo de CPU justa
    with torch.no_grad():
        _ = model(X_test_t[:5])
        functional.reset_net(model)

    # Medição de Tempo
    start_time = time.time()
    with torch.no_grad():
        preds = model(X_test_t)
        functional.reset_net(model)
    end_time = time.time()

    inference_time_ms = ((end_time - start_time) / len(y_test)) * 1000

    preds_np = preds.cpu().numpy().flatten()
    mae = mean_absolute_error(y_test, preds_np)
    rmse = np.sqrt(mean_squared_error(y_test, preds_np))
    r2 = r2_score(y_test, preds_np)

    return mae, rmse, r2, inference_time_ms


def main():
    config = PipelineConfig()
    set_seed(config.seed)
    logger = setup_logger("EdgeQuantization")
    os.makedirs(config.reports_dir, exist_ok=True)

    logger.info("=========================================================")
    logger.info("=== INICIANDO AVALIAÇÃO DE EDGE AI (PTQ / INT8) ===")
    logger.info("=========================================================")

    # 1. Carregar Dados de Teste
    X_test_t, y_test, n_features = get_holdout_test_data(config)
    logger.info(f"Dados do Cofre (Holdout) recriados. Amostras: {len(y_test)}")

    # 2. Instanciar o Modelo Original (FP32) a partir da Factory!
    # Isto garante que a arquitetura bate exatamente com o modelo treinado.
    models_dict = get_model_factory(config, n_features)
    model_name = "SJ-Spiking-MultiStep"

    if model_name not in models_dict:
        logger.error(f"O modelo {model_name} não foi encontrado na factory.")
        return

    model_fp32 = models_dict[model_name].to('cpu')

    model_path = os.path.join(config.holdout_models_dir, f"final_model_{model_name}.pth")
    if not os.path.exists(model_path):
        logger.error(f"Modelo não encontrado em {model_path}. Treine o Holdout primeiro.")
        return

    model_fp32.load_state_dict(torch.load(model_path, map_location='cpu'))
    size_fp32 = get_model_size_mb(model_fp32)
    logger.info(f"Modelo Float32 (Original) carregado com sucesso. Tamanho: {size_fp32:.4f} MB")

    # 3. Avaliar Modelo Original (FP32)
    logger.info("Executando inferência Float32...")
    mae_fp32, rmse_fp32, r2_fp32, time_fp32 = evaluate_inference(model_fp32, X_test_t, y_test)

    # 4. Aplicar Quantização (usando o common.py)
    dummy_input = X_test_t[:1]
    model_int8 = apply_post_training_quantization(model_fp32, dummy_input)
    size_int8 = get_model_size_mb(model_int8)

    # 5. Avaliar Modelo Quantizado (INT8)
    logger.info("Executando inferência Int8...")
    mae_int8, rmse_int8, r2_int8, time_int8 = evaluate_inference(model_int8, X_test_t, y_test)

    # 6. Relatório Final de Desempenho
    speedup = time_fp32 / time_int8 if time_int8 > 0 else 0

    logger.info("=========================================================")
    logger.info("=== RELATÓRIO DE DEPLOYMENT (BMS / MICROCONTROLADOR) ===")
    logger.info(
        f"-> Tamanho na Memória| FP32: {size_fp32:.4f} MB => INT8: {size_int8:.4f} MB ({(1 - size_int8 / size_fp32) * 100:.1f}% Redução)")
    logger.info(
        f"-> Precisão (MAE)    | FP32: {mae_fp32:.4f} => INT8: {mae_int8:.4f} (Diferença: {abs(mae_fp32 - mae_int8):.4f})")
    logger.info(f"-> Explicância (R²)  | FP32: {r2_fp32:.4f} => INT8: {r2_int8:.4f}")
    logger.info(f"-> Tempo p/ Amostra  | FP32: {time_fp32:.2f} ms => INT8: {time_int8:.2f} ms ({speedup:.2f}x)")

    # 7. Salvar resultados para CSV
    quant_data = [
        {'Format': 'FP32 (Original)', 'Size_MB': size_fp32, 'MAE': mae_fp32, 'RMSE': rmse_fp32, 'R2': r2_fp32,
         'Time_ms': time_fp32},
        {'Format': 'INT8 (Quantized)', 'Size_MB': size_int8, 'MAE': mae_int8, 'RMSE': rmse_int8, 'R2': r2_int8,
         'Time_ms': time_int8}
    ]
    csv_path = os.path.join(config.reports_dir, 'quantization_results.csv')
    pd.DataFrame(quant_data).to_csv(csv_path, index=False)
    logger.info(f"💾 Relatório de Quantização salvo em: {csv_path}")

    # 8. Salvar o modelo quantizado
    quantized_path = os.path.join(config.holdout_models_dir, f"quantized_{model_name}.pth")
    torch.save(model_int8.state_dict(), quantized_path)
    logger.info(f"💾 Modelo Int8 salvo para produção em: {quantized_path}")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()