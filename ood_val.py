import os
import torch
import numpy as np
import pandas as pd  # <--- ADICIONADO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from config.settings import PipelineConfig
from utils.common import setup_logger, set_seed
from data.ingestion import process_nasa_dataset
from features.builder import build_sequences
from models.factory import get_model_factory
from training.loops import train_standard, train_curriculum, train_spikingjelly, train_physics


def main():
    config = PipelineConfig()
    set_seed(config.seed)
    logger = setup_logger("OOD_Experiment")

    # Criar diretório de relatórios
    os.makedirs(config.reports_dir, exist_ok=True)

    logger.info("=========================================================")
    logger.info("=== EXPERIMENTO OOD (OUT-OF-DISTRIBUTION) NO FRIO (4°C) ===")
    logger.info("=========================================================")

    df = process_nasa_dataset(config)
    X, y, groups = build_sequences(df, config)
    n_features = X.shape[2]

    # SEPARAÇÃO OOD (Física)
    train_mask = np.array(['Frio' not in config.BATTERY_DOMAINS.get(bid, '') for bid in groups])
    test_mask = np.array(['Frio' in config.BATTERY_DOMAINS.get(bid, '') for bid in groups])

    X_train_val, y_train_val, groups_train_val = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test, groups_test = X[test_mask], y[test_mask], groups[test_mask]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.seed)
    t_idx, v_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))

    X_train, y_train = X_train_val[t_idx], y_train_val[t_idx]
    X_val, y_val = X_train_val[v_idx], y_train_val[v_idx]

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
    modelos_ood = {k: v for k, v in models_dict.items() if k in ["SJ-Spiking-MultiStep", "CNN-LSTM", "iTransformer"]}

    logger.info("\nIniciando Treinamento Zero-Shot OOD...")
    ood_results = {}
    ood_csv_data = []  # <--- LISTA PARA O CSV

    for name, model_instance in modelos_ood.items():
        logger.info(f"   ⏳ Treinando: {name} ...")

        if name.startswith("SJ-"):
            preds, _ = train_spikingjelly(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
        elif name == "Phys-iTR-Curriculum":
            preds, _ = train_curriculum(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
        elif name == "Phys-iTransformer":
            preds, _ = train_physics(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
        else:
            preds, _ = train_standard(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        ood_results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        logger.info(f"   ✓ OOD Test | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}\n")

        ood_csv_data.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

    # Guardar CSV do OOD
    pd.DataFrame(ood_csv_data).to_csv(os.path.join(config.reports_dir, 'ood_results.csv'), index=False)
    logger.info("💾 Relatório CSV do OOD salvo com sucesso.")

    logger.info("=== RESUMO DO EXPERIMENTO OOD (ZERO-SHOT NO FRIO) ===")
    for name, metrics in ood_results.items():
        logger.info(f"{name.ljust(25)} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()