import logging
import torch
import numpy as np
import os
import gc
import joblib
import json
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from models.factory import get_model_factory
from training.loops import train_standard, train_curriculum, train_spikingjelly, train_physics
from features.selection import select_best_features

logger = logging.getLogger("BatteryPipeline")


def run_cross_validation(X, y, groups, config):
    logger.info(f"Iniciando {config.k_folds}-Fold STRATIFIED Group Cross-Validation...")
    logger.info("⚠️  Feature selection será executada DENTRO de cada fold (apenas sobre X_train do fold).")
    os.makedirs(config.models_dir, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    # y_stratum usa todos os grupos (dimensão de X antes de qualquer slice de feature)
    y_stratum = np.array([config.BATTERY_DOMAINS.get(bid, 'Desconhecido') for bid in groups])

    cv_results = {}
    fold_data = {}
    histories_per_fold = {}
    best_snn_val_loss_overall = float('inf')
    fold_selected_features = {}   # auditoria: features escolhidas por fold

    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(X, y_stratum, groups)):
        current_fold = fold + 1
        logger.info(f"--- FOLD {current_fold}/{config.k_folds} ---")

        X_tv, y_tv, grp_tv = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed + fold)
        train_idx, val_idx = next(gss_val.split(X_tv, y_tv, grp_tv))

        X_train, y_train = X_tv[train_idx], y_tv[train_idx]
        X_val, y_val = X_tv[val_idx], y_tv[val_idx]

        # ── FEATURE SELECTION DENTRO DO FOLD ─────────────────────────────────
        # Calculada APENAS sobre X_train para evitar leakage do val e test sets.
        selected_idx, selected_names = select_best_features(
            X_dev=X_train,
            y_dev=y_train,
            feature_names=config.features,
            threshold=config.hi_correlation_threshold,
            method='spearman'
        )
        fold_selected_features[current_fold] = selected_names
        logger.info(f"   Features selecionadas no fold {current_fold}: {selected_names}")

        # Aplica o mesmo slice de features a train, val e test do fold
        X_train = X_train[:, :, selected_idx]
        X_val   = X_val[:, :, selected_idx]
        X_test  = X_test[:, :, selected_idx]
        n_features = X_train.shape[2]
        # ─────────────────────────────────────────────────────────────────────

        # ── Log de distribuição de domínios no fold ───────────────────────────
        train_domains = [config.BATTERY_DOMAINS.get(b, '?') for b in np.unique(grp_tv[train_idx])]
        val_domains   = [config.BATTERY_DOMAINS.get(b, '?') for b in np.unique(grp_tv[val_idx])]
        test_domains  = [config.BATTERY_DOMAINS.get(b, '?') for b in np.unique(groups[test_idx])]
        logger.info(f"   Domínios Treino : {sorted(train_domains)}")
        logger.info(f"   Domínios Val    : {sorted(val_domains)}")
        logger.info(f"   Domínios Teste  : {sorted(test_domains)}")
        # ─────────────────────────────────────────────────────────────────────

        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_val_s   = scaler_x.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test_s  = scaler_x.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_s   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        # Persiste scalers e features do fold para uso em quantize.py / análises externas
        joblib.dump(scaler_x, os.path.join(config.models_dir, f"scaler_x_fold{current_fold}.pkl"))
        joblib.dump(scaler_y, os.path.join(config.models_dir, f"scaler_y_fold{current_fold}.pkl"))
        with open(os.path.join(config.models_dir, f"selected_features_fold{current_fold}.json"), 'w') as fp:
            json.dump({'selected_indices': selected_idx, 'selected_features': list(selected_names)}, fp, indent=2)

        y_train_t = torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1).to(config.device)
        y_val_t   = torch.tensor(y_val_s,   dtype=torch.float32).unsqueeze(1).to(config.device)
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(config.device)
        X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32).to(config.device)
        X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32).to(config.device)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
        models_dict  = get_model_factory(config, n_features)

        fold_data[current_fold] = {'y_test': y_test, 'preds': {}}
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

            preds_physical = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
            mae  = mean_absolute_error(y_test, preds_physical)
            rmse = np.sqrt(mean_squared_error(y_test, preds_physical))
            r2   = r2_score(y_test, preds_physical)

            cv_results[name]['mae'].append(mae)
            cv_results[name]['rmse'].append(rmse)
            cv_results[name]['r2'].append(r2)
            fold_data[current_fold]['preds'][name] = preds_physical

            logger.info(f"   ✓ {name.ljust(25)} | Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}\n")

            torch.save(model_instance.state_dict(),
                       os.path.join(config.models_dir, f"{name}_fold{current_fold}.pth"))

            if name.startswith("SJ-"):
                min_val_loss_this_fold = min(history['val_loss'])
                if min_val_loss_this_fold < best_snn_val_loss_overall:
                    best_snn_val_loss_overall = min_val_loss_this_fold
                    fold_data['best_snn']      = model_instance
                    fold_data['best_snn_name'] = name
                    logger.info(f"      🏆 Novo melhor SNN global: {name} | Val Loss: {best_snn_val_loss_overall:.4f}")

        del train_loader, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, models_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Resumo de estabilidade das features seleccionadas por fold
    logger.info("=== ESTABILIDADE DA FEATURE SELECTION POR FOLD ===")
    for f, feats in fold_selected_features.items():
        logger.info(f"   Fold {f}: {feats}")

    return cv_results, fold_data, histories_per_fold