import logging

import numpy as np
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger("BatteryPipeline")


def select_best_features(X_dev, y_dev, feature_names, threshold=0.60, method='spearman'):
    """
    Avalia a correlação de cada feature com o Target (SoH).
    X_dev shape esperado: [Batch, Time, Features]
    """
    logger.info(f"=== SELEÇÃO DE FEATURES (HIs) via {method.upper()} ===")
    logger.info(f"Threshold mínimo de correlação: {threshold}")

    # Como os HIs são replicados ao longo do tempo (interpolação),
    # podemos pegar apenas o instante t=0 de cada ciclo para calcular a correlação.
    X_flat = X_dev[:, 0, :]

    selected_indices = []
    selected_names = []

    for i, name in enumerate(feature_names):
        feature_values = X_flat[:, i]

        if method == 'spearman':
            corr, p_value = spearmanr(feature_values, y_dev)
        else:
            corr, p_value = pearsonr(feature_values, y_dev)

        # Lidando com NaNs caso alguma feature tenha variância zero
        if np.isnan(corr):
            corr = 0.0

        # Verifica se passou no corte
        if abs(corr) >= threshold and p_value < 0.05:
            selected_indices.append(i)
            selected_names.append(name)
            logger.info(f"   ✓ [MANTIDA] {name.ljust(25)} | Corr: {corr:+.4f}")
        else:
            # Dica: Você pode querer forçar que features base (Voltage, Current, Temp)
            # nunca sejam descartadas, mesmo se a correlação for baixa.
            # Altere a linha 36 do selection.py para incluir o SoC:
            if name in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC']:
                selected_indices.append(i)
                selected_names.append(name)
                logger.info(f"   ! [FORÇADA] {name.ljust(25)} | Corr: {corr:+.4f} (Feature Base)")
            else:
                logger.warning(f"   x [DESCART] {name.ljust(25)} | Corr: {corr:+.4f}")

    logger.info(f"Features finais selecionadas ({len(selected_names)}): {selected_names}\n")
    return selected_indices, selected_names
