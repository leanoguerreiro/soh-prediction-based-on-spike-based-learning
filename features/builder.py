import numpy as np
import logging

logger = logging.getLogger("BatteryPipeline")


def build_sequences(df, config):
    logger.info("Construindo matrizes tridimensionais de features...")
    grouped = df.groupby(['battery_id', 'cycle_number'])
    seqs, soh, groups_list = [], [], []

    for (bid, cyc), g in grouped:
        if len(g) != config.time_steps:
            continue
        seqs.append(g[list(config.features)].values.astype(np.float32))
        soh.append(g['SoH'].iloc[0])
        groups_list.append(bid)

    X_seq = np.stack(seqs)
    y = np.array(soh).astype(np.float32)
    groups = np.array(groups_list)

    logger.info(f"Shape final: X={X_seq.shape}, y={y.shape}. Baterias únicas: {len(np.unique(groups))}")
    return X_seq, y, groups