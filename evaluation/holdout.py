import os
import torch
import logging
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

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

    # 2. Escalonamento (Fit apenas no Treino)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    def to_tensor(arr, tgt):
        return torch.tensor(arr, dtype=torch.float32).to(config.device), \
            torch.tensor(tgt, dtype=torch.float32).unsqueeze(1).to(config.device)

    X_train_t, y_train_t = to_tensor(X_train_s, y_train)
    X_val_t, y_val_t = to_tensor(X_val_s, y_val)
    X_test_t, _ = to_tensor(X_test_s, y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    models_dict = get_model_factory(config, n_features)

    holdout_results = {}
    holdout_histories = {}  # {model_name: history}

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

        # Métricas no conjunto de teste inédito
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        holdout_results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        logger.info(f"   ✓ Final {name.ljust(20)} | Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}\n")

        # SALVAMENTO FINAL
        save_path = os.path.join(config.holdout_models_dir, f"final_model_{name}.pth")
        torch.save(model_instance.state_dict(), save_path)
        logger.info(f"   💾 Modelo salvo em: {save_path}")

    return holdout_results, y_test, X_test_t, holdout_histories