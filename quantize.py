import copy
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from spikingjelly.activation_based import functional
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

warnings.filterwarnings("ignore", category=UserWarning, module="torchao")

from config.settings import PipelineConfig
from utils.common import setup_logger, set_seed
from data.ingestion import process_nasa_dataset
from features.builder import build_sequences
from models.factory import get_model_factory


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def get_model_size_mb(model):
    torch.save(model.state_dict(), "temp_size.p")
    size = os.path.getsize("temp_size.p") / 1e6
    os.remove("temp_size.p")
    return size


def apply_readout_only_quantization(model: nn.Module) -> nn.Module:
    """
    Quantiza apenas o readout (fusion/regressor/fc_out) via torchao,
    mantendo LIFNodes, atenções e projeções em FP32.
    """
    model_q = copy.deepcopy(model).to('cpu')
    model_q.eval()

    readout_attr = None
    for candidate in ('fusion', 'regressor', 'fc_out', 'readout'):
        if hasattr(model_q, candidate):
            readout_attr = candidate
            break

    if readout_attr is None:
        raise AttributeError(
            f"Nenhum readout encontrado em {type(model_q).__name__}. "
            "Esperado: 'fusion', 'regressor' ou 'fc_out'."
        )

    quantize_(getattr(model_q, readout_attr), Int8DynamicActivationInt8WeightConfig())
    return model_q


def evaluate_inference(model, X_test_t, y_test, scaler_y=None):
    """
    Avalia tempo de inferência e métricas.
    Se scaler_y for fornecido, des-escala as predições para a escala física.
    """
    model.eval()

    with torch.no_grad():
        _ = model(X_test_t[:5])
        functional.reset_net(model)

    start = time.time()
    with torch.no_grad():
        preds = model(X_test_t)
        functional.reset_net(model)
    elapsed = time.time() - start

    inference_time_ms = (elapsed / len(y_test)) * 1000
    preds_np = preds.cpu().numpy().flatten()

    if scaler_y is not None:
        preds_np = scaler_y.inverse_transform(preds_np.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test, preds_np)
    rmse = np.sqrt(mean_squared_error(y_test, preds_np))
    r2 = r2_score(y_test, preds_np)

    return mae, rmse, r2, inference_time_ms


def resolve_best_snn_name(config, models_dict, logger):
    """Identifica o melhor SNN via holdout_results.csv, com fallback para disco."""
    csv_path = os.path.join(config.reports_dir, 'holdout_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        snn_rows = df[df['Model'].str.startswith('SJ-')]
        if not snn_rows.empty:
            best = snn_rows.loc[snn_rows['MAE'].idxmin(), 'Model']
            logger.info(f"Melhor SNN via holdout_results.csv: {best}")
            return best

    for name in models_dict:
        if name.startswith("SJ-"):
            path = os.path.join(config.holdout_models_dir, f"final_model_{name}.pth")
            if os.path.exists(path):
                logger.warning(f"CSV não encontrado. Usando primeiro SNN disponível: {name}")
                return name
    return None


def load_model(model_instance, checkpoint_path, logger):
    """Carrega checkpoint com tratamento de erro."""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint não encontrado: {checkpoint_path}")
        return False
    model_instance.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model_instance.eval()
    return True


# ==============================================================================
# DADOS
# ==============================================================================

def get_holdout_test_data(config):
    """Recria X_holdout escalonado e carrega os scalers salvos pelo holdout.py."""
    import json
    df = process_nasa_dataset(config)
    X, y, groups = build_sequences(df, config)
    n_features_orig = X.shape[2]

    # Aplica seleção de features idêntica à do pipeline principal
    selected_features_path = os.path.join(config.reports_dir, 'selected_features.json')
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            sel = json.load(f)
        X = X[:, :, sel['selected_indices']]
        config.features = sel['selected_features']

    n_features = X.shape[2]

    gss_global = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    _, holdout_idx = next(gss_global.split(X, y, groups))
    X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

    scaler_x_path = os.path.join(config.holdout_models_dir, "scaler_x_final.pkl")
    scaler_y_path = os.path.join(config.holdout_models_dir, "scaler_y_final.pkl")

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError(
            f"Scalers não encontrados em {config.holdout_models_dir}. "
            "Execute main.py antes de quantize.py."
        )

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    X_test_s = scaler_x.transform(X_holdout.reshape(-1, n_features)).reshape(X_holdout.shape)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to('cpu')

    return X_test_t, y_holdout, n_features, scaler_y


def get_fold_test_data(config, X, y, groups, fold_num):
    """
    Recria o X_test e scaler_y de um fold específico da Cross-Validation,
    carregando o scaler_y salvo em disco para garantir consistência.
    Assume que X já foi filtrado pelas features selecionadas (feito em run_fold_quantization).
    """
    n_features = X.shape[2]
    y_stratum = np.array([config.BATTERY_DOMAINS.get(bid, 'Desconhecido') for bid in groups])

    sgkf = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    splits = list(sgkf.split(X, y_stratum, groups))

    train_val_idx, test_idx = splits[fold_num - 1]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler_y_path = os.path.join(config.models_dir, f"scaler_y_fold{fold_num}.pkl")
    scaler_x_path = os.path.join(config.models_dir, f"scaler_x_fold{fold_num}.pkl")

    if not os.path.exists(scaler_y_path) or not os.path.exists(scaler_x_path):
        raise FileNotFoundError(
            f"Scalers do fold {fold_num} não encontrados. "
            "Certifique-se de que cross_val.py salva scaler_x_fold{N}.pkl e scaler_y_fold{N}.pkl."
        )

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    X_test_s = scaler_x.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to('cpu')

    return X_test_t, y_test, scaler_y


# ==============================================================================
# AVALIAÇÃO HOLDOUT
# ==============================================================================

def run_holdout_quantization(config, logger):
    """Quantiza o melhor SNN do holdout e gera relatório."""
    logger.info("=== MODO: HOLDOUT ===")

    X_test_t, y_test, n_features, scaler_y = get_holdout_test_data(config)
    logger.info(f"Amostras Holdout: {len(y_test)}")

    models_dict = get_model_factory(config, n_features)
    model_name = resolve_best_snn_name(config, models_dict, logger)
    if model_name is None:
        logger.error("Nenhum SNN encontrado. Execute main.py primeiro.")
        return []

    model_fp32 = models_dict[model_name].to('cpu')
    ckpt_path = os.path.join(config.holdout_models_dir, f"final_model_{model_name}.pth")
    if not load_model(model_fp32, ckpt_path, logger):
        return []

    size_fp32 = get_model_size_mb(model_fp32)
    logger.info(f"Modelo {model_name} carregado | FP32: {size_fp32:.4f} MB")

    mae_fp32, rmse_fp32, r2_fp32, time_fp32 = evaluate_inference(
        model_fp32, X_test_t, y_test, scaler_y
    )

    model_int8 = apply_readout_only_quantization(model_fp32)
    size_int8 = get_model_size_mb(model_int8)

    mae_int8, rmse_int8, r2_int8, time_int8 = evaluate_inference(
        model_int8, X_test_t, y_test, scaler_y
    )

    speedup = time_fp32 / time_int8 if time_int8 > 0 else 0
    logger.info(
        f"-> Tamanho  | FP32: {size_fp32:.4f} MB => INT8: {size_int8:.4f} MB ({(1 - size_int8 / size_fp32) * 100:.1f}% redução)")
    logger.info(f"-> MAE      | FP32: {mae_fp32:.4f} => INT8: {mae_int8:.4f} (Δ {abs(mae_fp32 - mae_int8):.4f})")
    logger.info(f"-> R²       | FP32: {r2_fp32:.4f} => INT8: {r2_int8:.4f}")
    logger.info(f"-> Tempo    | FP32: {time_fp32:.2f} ms => INT8: {time_int8:.2f} ms ({speedup:.2f}x)")

    torch.save(model_int8.state_dict(),
               os.path.join(config.holdout_models_dir, f"quantized_{model_name}.pth"))
    logger.info(f"💾 Modelo INT8 salvo: quantized_{model_name}.pth")

    return [
        {'Source': 'holdout', 'Fold': '-', 'Model': model_name,
         'Format': 'FP32', 'Size_MB': size_fp32, 'MAE': mae_fp32,
         'RMSE': rmse_fp32, 'R2': r2_fp32, 'Time_ms': time_fp32},
        {'Source': 'holdout', 'Fold': '-', 'Model': model_name,
         'Format': 'INT8', 'Size_MB': size_int8, 'MAE': mae_int8,
         'RMSE': rmse_int8, 'R2': r2_int8, 'Time_ms': time_int8},
    ]


# ==============================================================================
# AVALIAÇÃO POR FOLD
# ==============================================================================

def run_fold_quantization(config, logger):
    """
    Para cada fold e cada arquitetura SNN, carrega o checkpoint treinado,
    quantiza o readout e compara FP32 vs INT8 no conjunto de teste daquele fold.
    """
    logger.info("=== MODO: CROSS-VALIDATION (por fold) ===")

    import json
    df_raw = process_nasa_dataset(config)
    X, y, groups = build_sequences(df_raw, config)

    # Aplica seleção de features idêntica à do pipeline principal
    selected_features_path = os.path.join(config.reports_dir, 'selected_features.json')
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            sel = json.load(f)
        X = X[:, :, sel['selected_indices']]
        config.features = sel['selected_features']
        logger.info(f"Features selecionadas carregadas: {config.features}")
    else:
        logger.warning("selected_features.json não encontrado. Usando todas as features.")

    n_features = X.shape[2]

    models_dict = get_model_factory(config, n_features)
    snn_names = [n for n in models_dict if n.startswith("SJ-")]

    results = []

    for fold_num in range(1, config.k_folds + 1):
        logger.info(f"--- FOLD {fold_num}/{config.k_folds} ---")

        try:
            X_test_t, y_test, scaler_y = get_fold_test_data(config, X, y, groups, fold_num)
        except FileNotFoundError as e:
            logger.error(str(e))
            break

        for model_name in snn_names:
            ckpt_path = os.path.join(config.models_dir, f"{model_name}_fold{fold_num}.pth")

            # Instancia uma cópia limpa da arquitetura
            model_fp32 = get_model_factory(config, n_features)[model_name].to('cpu')
            if not load_model(model_fp32, ckpt_path, logger):
                continue

            size_fp32 = get_model_size_mb(model_fp32)
            mae_fp32, rmse_fp32, r2_fp32, time_fp32 = evaluate_inference(
                model_fp32, X_test_t, y_test, scaler_y
            )

            model_int8 = apply_readout_only_quantization(model_fp32)
            size_int8 = get_model_size_mb(model_int8)
            mae_int8, rmse_int8, r2_int8, time_int8 = evaluate_inference(
                model_int8, X_test_t, y_test, scaler_y
            )

            speedup = time_fp32 / time_int8 if time_int8 > 0 else 0
            logger.info(
                f"   {model_name.ljust(25)} Fold {fold_num} | "
                f"MAE FP32: {mae_fp32:.4f} => INT8: {mae_int8:.4f} | "
                f"R² FP32: {r2_fp32:.4f} => INT8: {r2_int8:.4f} | "
                f"{speedup:.2f}x"
            )

            # Salva modelo INT8 por fold
            torch.save(
                model_int8.state_dict(),
                os.path.join(config.models_dir, f"quantized_{model_name}_fold{fold_num}.pth")
            )

            for fmt, vals in [
                ('FP32', (size_fp32, mae_fp32, rmse_fp32, r2_fp32, time_fp32)),
                ('INT8', (size_int8, mae_int8, rmse_int8, r2_int8, time_int8)),
            ]:
                results.append({
                    'Source': 'cv', 'Fold': fold_num, 'Model': model_name,
                    'Format': fmt, 'Size_MB': vals[0], 'MAE': vals[1],
                    'RMSE': vals[2], 'R2': vals[3], 'Time_ms': vals[4],
                })

    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    config = PipelineConfig()
    set_seed(config.seed)
    logger = setup_logger("EdgeQuantization")
    os.makedirs(config.reports_dir, exist_ok=True)

    logger.info("=== INICIANDO AVALIAÇÃO DE EDGE AI (PTQ / INT8 — torchao) ===")

    all_results = []

    # --- Holdout ---
    all_results.extend(run_holdout_quantization(config, logger))

    # --- Cross-Validation por fold ---
    all_results.extend(run_fold_quantization(config, logger))

    # --- CSV consolidado ---
    if all_results:
        csv_path = os.path.join(config.reports_dir, 'quantization_results.csv')
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 CSV consolidado salvo em: {csv_path}")

        # Resumo de estabilidade: média e std da degradação de MAE por modelo
        logger.info("=== ESTABILIDADE DA QUANTIZAÇÃO POR MODELO (CV) ===")
        cv_df = df[df['Source'] == 'cv'].copy()
        if not cv_df.empty:
            for model_name in cv_df['Model'].unique():
                fp32 = cv_df[(cv_df['Model'] == model_name) & (cv_df['Format'] == 'FP32')]['MAE']
                int8 = cv_df[(cv_df['Model'] == model_name) & (cv_df['Format'] == 'INT8')]['MAE']
                delta = (int8.values - fp32.values)
                logger.info(
                    f"   {model_name.ljust(25)} | "
                    f"ΔMAE médio: {delta.mean():+.4f} ± {delta.std():.4f} | "
                    f"Speedup médio: {(cv_df[(cv_df['Model'] == model_name) & (cv_df['Format'] == 'FP32')]['Time_ms'].values / cv_df[(cv_df['Model'] == model_name) & (cv_df['Format'] == 'INT8')]['Time_ms'].values).mean():.2f}x"
                )

    logger.info("=== QUANTIZAÇÃO CONCLUÍDA ===")


if __name__ == "__main__":
    main()
