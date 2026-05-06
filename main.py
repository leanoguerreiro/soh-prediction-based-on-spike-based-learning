import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from config.settings import PipelineConfig
from data.ingestion import process_nasa_dataset
from data.ingestion_synthetic import process_synthetic_dataset
from evaluation.cross_val import run_cross_validation
from evaluation.holdout import run_holdout_evaluation
from features.builder import build_sequences
from features.selection import select_best_features
from utils.common import setup_logger, set_seed
from visualization.plots import generate_all_plots, plot_loss_curves, generate_holdout_plots


def main():
    # 1. Inicialização e Configuração
    config = PipelineConfig()
    set_seed(config.seed)
    logger = setup_logger()

    # Criar diretório de relatórios
    os.makedirs(config.reports_dir, exist_ok=True)

    logger.info("=== INICIANDO PIPELINE DE BATERIAS (SOH) ===")
    logger.info(f"Device: {config.device} | Epochs: {config.epochs} | K-Folds: {config.k_folds}")

    # 2. Ingestão de Dados e Construção de Features
    if config.use_synthetic_data:
        logger.info("Carregando dados sintéticos...")
        df = process_synthetic_dataset(config)
    else:
        logger.info("Carregando dados reais (NASA)...")
        df = process_nasa_dataset(config)
    X, y, groups = build_sequences(df, config)

    # O COFRE: SEPARAÇÃO GLOBAL DO HOLDOUT
    logger.info("Separando conjunto Holdout Global (Cofre)...")
    gss_global = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    dev_idx, holdout_idx = next(gss_global.split(X, y, groups))

    X_dev, y_dev, groups_dev = X[dev_idx], y[dev_idx], groups[dev_idx]
    X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

    # ==========================================================================
    # 3. SELEÇÃO DE HEALTH INDICATORS (NOVA ETAPA)
    # ==========================================================================
    # Rodamos a correlação APENAS no X_dev para evitar Data Leakage do cofre
    selected_idx, selected_features = select_best_features(
        X_dev=X_dev,
        y_dev=y_dev,
        feature_names=config.features,
        threshold=config.hi_correlation_threshold,  # vem do PipelineConfig, não hardcoded
        method='spearman'
    )
    config.features = selected_features  # atualiza features na config

    # Persiste as features selecionadas para reprodutibilidade do holdout standalone
    import json as _json
    selected_features_path = os.path.join(config.reports_dir, 'selected_features.json')
    with open(selected_features_path, 'w') as _f:
        _json.dump({'selected_indices': selected_idx, 'selected_features': list(selected_features)}, _f, indent=2)
    logger.info(f"💾 Features selecionadas salvas em: {selected_features_path}")

    X_dev = X_dev[:, :, selected_idx]
    X_holdout = X_holdout[:, :, selected_idx]

    # ==========================================================================
    # 4A. CROSS-VALIDATION
    # ==========================================================================
    logger.info("=== ETAPA: CROSS-VALIDATION ===")
    cv_results, fold_data, histories_per_fold = run_cross_validation(X_dev, y_dev, groups_dev, config)

    logger.info("=== RESULTADOS GLOBAIS — CROSS-VALIDATION ===")
    cv_summary_data = []
    cv_detailed_data = []

    for model_name in cv_results.keys():
        mae_mean, mae_std = np.mean(cv_results[model_name]['mae']), np.std(cv_results[model_name]['mae'])
        rmse_mean, rmse_std = np.mean(cv_results[model_name]['rmse']), np.std(cv_results[model_name]['rmse'])
        r2_mean, r2_std = np.mean(cv_results[model_name]['r2']), np.std(cv_results[model_name]['r2'])

        logger.info(
            f"{model_name.ljust(25)} | "
            f"MAE: {mae_mean:.4f} ± {mae_std:.4f} | "
            f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f} | "
            f"R²: {r2_mean:.4f} ± {r2_std:.4f}"
        )

        # Preparar dados para CSV: Resumo
        cv_summary_data.append({
            'Model': model_name,
            'MAE_Mean': mae_mean, 'MAE_Std': mae_std,
            'RMSE_Mean': rmse_mean, 'RMSE_Std': rmse_std,
            'R2_Mean': r2_mean, 'R2_Std': r2_std
        })

        # Preparar dados para CSV: Detalhado por Fold
        for fold_idx in range(config.k_folds):
            cv_detailed_data.append({
                'Model': model_name,
                'Fold': fold_idx + 1,
                'MAE': cv_results[model_name]['mae'][fold_idx],
                'RMSE': cv_results[model_name]['rmse'][fold_idx],
                'R2': cv_results[model_name]['r2'][fold_idx]
            })

    # Guardar CSVs do Cross-Validation
    pd.DataFrame(cv_summary_data).to_csv(os.path.join(config.reports_dir, 'cv_summary_results.csv'), index=False)
    pd.DataFrame(cv_detailed_data).to_csv(os.path.join(config.reports_dir, 'cv_detailed_results.csv'), index=False)
    logger.info("💾 Relatórios CSV do Cross-Validation salvos com sucesso.")

    # ==========================================================================
    # 4B. HOLDOUT
    # ==========================================================================
    logger.info("=== ETAPA: HOLDOUT ===")
    holdout_results, y_test, X_test_t, holdout_histories, holdout_preds = run_holdout_evaluation(
        X_dev, y_dev, groups_dev, X_holdout, y_holdout, config
    )

    logger.info("=== RESULTADOS GLOBAIS — HOLDOUT ===")
    holdout_data = []
    for model_name, metrics in holdout_results.items():
        logger.info(
            f"{model_name.ljust(25)} | "
            f"MAE: {metrics['mae']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | "
            f"R²: {metrics['r2']:.4f}"
        )
        holdout_data.append({
            'Model': model_name,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R2': metrics['r2']
        })

    # Guardar CSV do Holdout
    pd.DataFrame(holdout_data).to_csv(os.path.join(config.reports_dir, 'holdout_results.csv'), index=False)
    logger.info("💾 Relatório CSV do Holdout salvo com sucesso.")

    # ==========================================================================
    # 5. Quantização e Visualizações
    # ==========================================================================
    if 'best_snn' in fold_data:
        torch.save(fold_data['best_snn'].state_dict(),
                   os.path.join(config.models_dir, f"best_snn_fold_{fold_data['best_snn_name']}.pth"))

    logger.info("=== ETAPA: VISUALIZAÇÕES ===")
    plot_loss_curves(histories_per_fold, save_dir=config.plots_cv_dir)
    plot_loss_curves(holdout_histories, fold=1, save_dir=config.plots_holdout_dir)
    generate_all_plots(cv_results, fold_data, save_dir=config.plots_cv_dir)
    generate_holdout_plots(y_test, holdout_preds, save_dir=config.plots_holdout_dir)

    logger.info("Pipeline executado com sucesso.")

    # Serializa o estado final da config (features selecionadas incluídas)
    config.save(os.path.join(config.reports_dir, 'pipeline_config_final.json'))
    logger.info("💾 Config final serializada com sucesso.")


if __name__ == "__main__":
    main()
