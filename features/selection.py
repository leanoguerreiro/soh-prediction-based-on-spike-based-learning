import logging
import numpy as np
import polars as pl

logger = logging.getLogger("BatteryPipeline")

def select_best_features(X_dev, y_dev, feature_names, threshold=0.60, method='pearson'):
    """
    Avalia a correlação de cada feature com o Target (SoH) usando Polars.
    X_dev shape esperado: [Batch, Time, Features]
    """
    logger.info(f"=== SELEÇÃO DE FEATURES (HIs) via {method.upper()} (Polars) ===")
    logger.info(f"Threshold mínimo de correlação: {threshold}")

    # Como os HIs são replicados ao longo do tempo (interpolação),
    # pegamos apenas o instante t=0 de cada ciclo para calcular a correlação.
    X_flat = X_dev[:, 0, :]

    # Injeta a matriz NumPy para dentro do Polars para processamento massivo em paralelo
    df_corr = pl.DataFrame(X_flat, schema=list(feature_names))
    df_corr = df_corr.with_columns(pl.Series("Target_SoH", y_dev))

    selected_indices = []
    selected_names = []

    for i, name in enumerate(feature_names):
        # O Polars faz o cálculo de correlação diretamente em memória
        corr = df_corr.select(
            pl.corr(name, "Target_SoH", method=method)
        ).item()

        # Lidando com features de variância zero (como o SoC no t=0)
        if corr is None or np.isnan(corr):
            corr = 0.0

        # Verifica se passou no corte absoluto (p-value ignorado pois N amostras é gigante)
        if abs(corr) >= threshold:
            selected_indices.append(i)
            selected_names.append(name)
            logger.info(f"   ✓ [MANTIDA] {name.ljust(25)} | Corr: {corr:+.4f}")
        else:
            # Proteção das features de leitura crua do hardware
            if name in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC']:
                selected_indices.append(i)
                selected_names.append(name)
                logger.info(f"   ! [FORÇADA] {name.ljust(25)} | Corr: {corr:+.4f} (Feature Base)")
            else:
                logger.warning(f"   x [DESCART] {name.ljust(25)} | Corr: {corr:+.4f}")

    logger.info(f"Features finais selecionadas ({len(selected_names)}): {selected_names}\n")
    return selected_indices, selected_names