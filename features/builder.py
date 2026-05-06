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

    leakage_features = ["Capacity_Ah", "Energy_Wh", "Delta_Q"]
    features_input = [f for f in config.features if f not in leakage_features]

    agg_df = df.group_by(["battery_id", "cycle_number"], maintain_order=True).agg([
        pl.len().alias("count"),
        pl.col("SoH").first().alias("SoH"),
        pl.col("Capacity_Ah").first().alias("Capacity_Ah") if "Capacity_Ah" in df.columns else None,
        *[pl.col(f) for f in features_input]
    ]).filter(pl.col("count") == config.time_steps)

    # 2. Constrói a Matriz X apenas com sensores e HIs térmicos/temporais
    X_seq = np.stack([
        np.vstack(agg_df[f].to_list())
        for f in features_input], axis=-1).astype(np.float32)

    y = agg_df["SoH"].to_numpy().astype(np.float32)
    groups = agg_df["battery_id"].to_numpy()

    logger.info(f"Shape final: X={X_seq.shape}, y={y.shape}. Baterias únicas: {len(np.unique(groups))}")
    return X_seq, y, groups