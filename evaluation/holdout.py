import os
import torch
import logging
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import joblib

from models.factory import get_model_factory
from training.loops import (
    train_standard, train_curriculum,
    train_spikingjelly, train_physics
)

logger = logging.getLogger("BatteryPipeline")


def run_holdout_evaluation(X_dev, y_dev, groups_dev, X_holdout, y_holdout, config):
    logger.info("Iniciando Treinamento Final e Avaliação Holdout...")
    os.makedirs(config.holdout_models_dir, exist_ok=True)

    n_features = X_dev.shape[2]

    # 1. Divisão: Separação de VALIDAÇÃO a partir do Dev Set
    gss_val = GroupShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.seed)
    train_idx, val_idx = next(gss_val.split(X_dev, y_dev, groups_dev))

    X_train, y_train = X_dev[train_idx], y_dev[train_idx]
    X_val, y_val = X_dev[val_idx], y_dev[val_idx]

    # O X_test e y_test agora vêm diretamente do cofre
    X_test, y_test = X_holdout, y_holdout

    # 1. Escalonamento do X (Já estava correto)
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = scaler_x.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_s = scaler_x.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # 2. Escalonamento do Y (A SOLUÇÃO)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # Note que não escalonamos o y_test para as métricas finais, apenas os de treino/validação
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1).to(config.device)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32).unsqueeze(1).to(config.device)
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(config.device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(config.device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(config.device)

    scaler_x_path = os.path.join(config.holdout_models_dir, "scaler_x_final.pkl")
    joblib.dump(scaler_x, scaler_x_path)
    logger.info(f"   💾 Scaler_x salvo em: {scaler_x_path}")

    scaler_y_path = os.path.join(config.holdout_models_dir, "scaler_y_final.pkl")
    joblib.dump(scaler_y, scaler_y_path)
    logger.info(f"   💾 Scaler_y salvo em: {scaler_y_path}")

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    models_dict = get_model_factory(config, n_features)

    holdout_results = {}
    holdout_histories = {}  # {model_name: history}
    holdout_preds = {}

    # 4. Loop de Treinamento e Salvamento
    for name, model_instance in models_dict.items():
        logger.info(f"   ⏳ Treinando modelo final: {name} ...")

        if name.startswith("SJ-"):
            preds, history = train_spikingjelly(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
        elif name == "Phys-iTR-Curriculum":
            preds, history = train_curriculum(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
        elif name == "Phys-iTransformer":
            preds, history = train_physics(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
        else:
            preds, history = train_standard(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)

        holdout_histories[name] = history

        # DES-ESCALONAMENTO: Transformar predição [0, 1] de volta para escala física (ex: 70-100)
        preds_physical = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

        # Cálculo de métricas sobre valores reais
        mae = mean_absolute_error(y_test, preds_physical)
        rmse = np.sqrt(mean_squared_error(y_test, preds_physical))
        r2 = r2_score(y_test, preds_physical)

        holdout_results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        holdout_histories[name] = history
        holdout_preds[name] = preds_physical

        logger.info(f"   ✓ {name.ljust(25)} | MAE: {mae:.4f} | R²: {r2:.4f}")

        # Salvar checkpoint final
        torch.save(model_instance.state_dict(), os.path.join(config.holdout_models_dir, f"final_model_{name}.pth"))

    return holdout_results, y_test, X_test_t, holdout_histories, holdout_preds