import logging
import numpy as np
import polars as pl
import pandas as pd

logger = logging.getLogger("BatteryPipeline")


def build_sequences(df, config):
    logger.info("Construindo matrizes tridimensionais de features com Polars...")

    # 1. Garante que é um Polars DataFrame (caso venha do return .to_pandas() do ingestion.py)
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    features = list(config.features)

    # 2. Agrupamento ultra-rápido em C/Rust.
    # O maintain_order=True é CRUCIAL para garantir que a linha do tempo (Time) não fica baralhada
    agg_df = df.group_by(["battery_id", "cycle_number"], maintain_order=True).agg([
        pl.len().alias("count"),
        pl.col("SoH").first().alias("SoH"),
        *[pl.col(f) for f in features]
    ]).filter(pl.col("count") == config.time_steps)

    # 3. Construção Vetorizada do Tensor 3D (Batch, Time, Features)
    # agg_df[f].to_list() devolve as sequências temporais;
    # np.vstack converte as listas num array 2D;
    # np.stack(..., axis=-1) sobrepõe as features formando a 3ª dimensão.
    X_seq = np.stack([
        np.vstack(agg_df[f].to_list())
        for f in features], axis=-1).astype(np.float32)

    # 4. Extração simples das variáveis alvo (Target)
    y = agg_df["SoH"].to_numpy().astype(np.float32)
    groups = agg_df["battery_id"].to_numpy()

    logger.info(f"Shape final: X={X_seq.shape}, y={y.shape}. Baterias únicas: {len(np.unique(groups))}")
    return X_seq, y, groups