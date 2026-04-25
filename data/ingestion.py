import os
import polars as pl
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import logging

logger = logging.getLogger("BatteryPipeline")


def process_nasa_dataset(config):
    logger.info("Iniciando ingestão e processamento do Dataset NASA com Polars...")
    os.makedirs(config.output_dir, exist_ok=True)

    metadata_path = os.path.join(config.input_dir, "metadata.csv")

    # 1. Leitura do metadata.csv usando Polars (ignorando os números complexos da impedância)
    metadata = pl.read_csv(metadata_path, ignore_errors=True).with_columns(
        pl.col('battery_id').cast(pl.Utf8)
    )

    # 2. Filtragem e cálculo do 'cycle_number' de forma vetorial
    discharge_metadata = metadata.filter(
        (pl.col('type') == 'discharge') &
        (~pl.col('battery_id').is_in(config.excluded_batteries))
    ).with_columns(
        # Cria uma contagem cumulativa começando em 1 para cada grupo de battery_id
        pl.int_range(1, pl.len() + 1).over('battery_id').alias('cycle_number')
    )

    processed_dfs = []
    missing_files = set()

    # iter_rows(named=True) devolve um dicionário para cada linha, super rápido
    for row in tqdm(discharge_metadata.iter_rows(named=True), total=discharge_metadata.height,
                    desc="Processando Ciclos"):
        file_path = os.path.join(config.input_dir, "data", row['filename'])
        if not os.path.exists(file_path):
            missing_files.add(row['battery_id'])
            continue

        # Leitura ultra-rápida do CSV do ciclo
        df = pl.read_csv(file_path)

        # 3. Cutoff - Cortar o DataFrame quando a voltagem cai abaixo de 2.7V
        cutoff_mask = df["Voltage_measured"] < 2.7
        if cutoff_mask.any():
            cutoff_idx = cutoff_mask.arg_true()[0]  # Pega no primeiro índice verdadeiro
            df = df.slice(0, cutoff_idx)

        # 4. Cálculo da Capacidade (Delta_Q e Time_diff) em Rust (motor Polars)
        df = df.with_columns(
            (pl.col('Time').diff().fill_null(0) / 3600).alias('Time_diff_hr')
        ).with_columns(
            (pl.col('Current_measured') * pl.col('Time_diff_hr')).alias('Delta_Q')
        )

        capacity = abs(df["Delta_Q"].sum())

        if capacity > 1.4:
            # Cálculo vetorial do SoC
            df = df.with_columns(
                (100 * (1 + pl.col('Delta_Q').cum_sum() / capacity)).alias('SoC')
            )
            soh_value = (capacity / 2.0) * 100

            # 5. Filtragem da Janela de Observação (Coach B)
            start_time = df["Time"][0]
            end_time_coach_b = start_time + config.observation_window_sec

            coach_b_df = df.filter(pl.col('Time') <= end_time_coach_b)

            if coach_b_df.height < 5:
                continue

            # Extração para Numpy (Zero-Copy) para acelerar cálculos do Numpy/Scipy
            time_arr = coach_b_df["Time"].to_numpy()
            volt_arr = coach_b_df["Voltage_measured"].to_numpy()
            temp_arr = coach_b_df["Temperature_measured"].to_numpy()

            # =====================================================================
            # CÁLCULO DOS HEALTH INDICATORS (HIs)
            # =====================================================================
            _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
            hi_values = {
                'HI_Time_of_Discharge': time_arr[-1] - time_arr[0],
                'HI_Max_Temp': temp_arr.max(),
                'HI_Voltage_Integral': _trapz(volt_arr, time_arr),
                'HI_Voltage_Drop': volt_arr.max() - volt_arr.min()
            }
            # =====================================================================

            uniform_time_axis = np.linspace(start_time, end_time_coach_b, config.time_steps)
            interpolated_cycle = {
                'battery_id': [row['battery_id']] * config.time_steps,
                'cycle_number': [row['cycle_number']] * config.time_steps,
                'SoH': [soh_value] * config.time_steps,
            }

            # Lógica de Interpolação vs Repetição
            for feat in config.features:
                if feat.startswith('HI_'):
                    val = hi_values.get(feat, 0.0)
                    interpolated_cycle[feat] = [val] * config.time_steps
                else:
                    feat_arr = coach_b_df[feat].to_numpy()
                    f_interp = interp1d(time_arr, feat_arr, kind='linear', fill_value='extrapolate')
                    interpolated_cycle[feat] = f_interp(uniform_time_axis)

            # Usamos o Polars para construir o pequeno DF interpolado
            processed_dfs.append(pl.DataFrame(interpolated_cycle))

    # 6. Concatenação de todos os DataFrames interpolados num piscar de olhos
    full_dataset_pl = pl.concat(processed_dfs)

    # Logs de auditoria originais
    if missing_files:
        logger.warning(f"⚠️ Ficheiros não encontrados para {len(missing_files)} baterias: {sorted(missing_files)}")

    unique_batteries = full_dataset_pl["battery_id"].unique().to_list()
    mapped_not_processed = set(config.BATTERY_DOMAINS.keys()) - set(unique_batteries)

    if mapped_not_processed:
        logger.warning(
            f"⚠️ Baterias definidas em BATTERY_DOMAINS mas ausentes do dataset processado "
            f"(verifique ficheiros ou excluded_batteries): {sorted(mapped_not_processed)}"
        )

    csv_data_path = os.path.join(config.output_dir, 'battery_health_dataset.csv')
    full_dataset_pl.write_csv(csv_data_path)
    logger.info(f"Dataset salvo em {csv_data_path}")

    # CONVERSÃO PARA PANDAS NO FINAL
    # Retornamos como Pandas para garantir compatibilidade a 100% com o builder.py
    return full_dataset_pl