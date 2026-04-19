import logging
import torch
import numpy as np
import os
import gc
# 1. Altere a importação para StratifiedGroupKFold
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from models.factory import get_model_factory
from training.loops import train_standard, train_curriculum, train_spikingjelly, train_physics

logger = logging.getLogger("BatteryPipeline")


def run_cross_validation(X, y, groups, config):
    logger.info(f"Iniciando {config.k_folds}-Fold STRATIFIED Group Cross-Validation...")
    os.makedirs(config.models_dir, exist_ok=True)

    # 3. Inicializa o Stratified K-Fold
    sgkf = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    n_features = X.shape[2]

    # 4. Cria o array 'y_stratum' (as classes físicas para estratificação)
    # Como o 'groups' tem o ID da bateria, mapeamos para a string do domínio físico
    y_stratum = np.array([config.BATTERY_DOMAINS.get(bid, 'Desconhecido') for bid in groups])

    cv_results = {}
    fold_data = {}
    histories_per_fold = {}

    # 5. Passa o y_stratum para a função split balancear os Folds
    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(X, y_stratum, groups)):
        current_fold = fold + 1
        logger.info(f"--- FOLD {current_fold}/{config.k_folds} ---")

        X_tv, y_tv, grp_tv = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        train_idx, val_idx = next(gss_val.split(X_tv, y_tv, grp_tv))

        X_train, y_train = X_tv[train_idx], y_tv[train_idx]
        X_val, y_val = X_tv[val_idx], y_tv[val_idx]

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

        fold_data[current_fold] = {'y_test': y_test, 'preds': {}, 'X_test_t': X_test_t}
        histories_per_fold[current_fold] = {}

        for name, model_instance in models_dict.items():
            if name not in cv_results:
                cv_results[name] = {'mae': [], 'rmse': [], 'r2': []}

            logger.info(f"   ⏳ Iniciando treino do modelo: {name} ...")

            if name.startswith("SJ-"):
                preds, history = train_spikingjelly(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
            elif name == "Phys-iTR-Curriculum":
                preds, history = train_curriculum(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
            elif name == "Phys-iTransformer":
                preds, history = train_physics(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)
            else:
                preds, history = train_standard(model_instance, train_loader, X_val_t, y_val_t, X_test_t, config)

            histories_per_fold[current_fold][name] = history

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            cv_results[name]['mae'].append(mae)
            cv_results[name]['rmse'].append(rmse)
            cv_results[name]['r2'].append(r2)
            fold_data[current_fold]['preds'][name] = preds

            logger.info(f"   ✓ {name.ljust(25)} | Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}\n")

            model_filename = f"{name}_fold{current_fold}.pth"
            model_path = os.path.join(config.models_dir, model_filename)
            torch.save(model_instance.state_dict(), model_path)

            # Salvar último modelo para PTQ
            if current_fold == config.k_folds and name == "SJ-Spiking-MultiStep":
                fold_data['best_snn'] = model_instance

        # 1. Deleta as referências aos loaders e dicionários do fold atual
        del train_loader
        del X_train_t, y_train_t, X_val_t, y_val_t, X_test_t
        del models_dict

        # 2. Força o Garbage Collector do Python a agir AGORA
        gc.collect()

        # 3. Esvazia o cache reservado do PyTorch na Placa de Vídeo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cv_results, fold_data, histories_per_fold