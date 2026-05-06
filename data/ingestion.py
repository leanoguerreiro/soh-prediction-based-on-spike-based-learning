import os
import polars as pl
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger("BatteryPipeline")


def process_single_cycle(row, config):
    """Processa um único ciclo da bateria."""
    file_path = os.path.join(config.input_dir, "data", row['filename'])
    if not os.path.exists(file_path):
        return None

    df = pl.read_csv(file_path)

    # Cutoff (Voltagem < 2.7V)
    cutoff_mask = df["Voltage_measured"] < 2.7
    if cutoff_mask.any():
        cutoff_idx = cutoff_mask.arg_true()[0]
        df = df.slice(0, cutoff_idx)

    # Cálculo da Capacidade (Delta_Q) e Potência/Energia
    df = df.with_columns(
        (pl.col('Time').diff().fill_null(0) / 3600).alias('Time_diff_hr')
    ).with_columns(
        (pl.col('Current_measured') * pl.col('Time_diff_hr')).alias('Delta_Q'),
        (pl.col('Voltage_measured') * pl.col('Current_measured') * pl.col('Time_diff_hr')).alias('Energy_Wh')
    )

    capacity = abs(df["Delta_Q"].sum())
    total_energy_wh = abs(df["Energy_Wh"].sum())  # Nova variável calculada!

    if capacity < 0.1:
        return None

    # Cálculo do SoH dinâmico (com base na capacidade nominal)
    c_nom = 2.0
    soh_value = (capacity / c_nom) * 100

    df = df.with_columns(
        (100 * (1 + pl.col('Delta_Q').cum_sum() / capacity)).alias('SoC')
    )

    start_time = df["Time"][0]
    end_time_coach_b = start_time + config.observation_window_sec
    coach_b_df = df.filter(pl.col('Time') <= end_time_coach_b)

    if coach_b_df.height < 5:
        return None

    time_arr = coach_b_df["Time"].to_numpy()
    volt_arr = coach_b_df["Voltage_measured"].to_numpy()
    temp_arr = coach_b_df["Temperature_measured"].to_numpy()


    hi_values = {
        'HI_Time_of_Discharge': time_arr[-1] - time_arr[0],
        'HI_Max_Temp': temp_arr.max(),
        'HI_Temp_Delta': temp_arr.max() - temp_arr.min(),  # Novo HI
        'HI_Mean_Temp': temp_arr.mean(),  # Novo HI
        'HI_Voltage_Integral': np.trapezoid(volt_arr, time_arr),
        'HI_Voltage_Drop': volt_arr.max() - volt_arr.min()
    }

    # Interpolação Linear Otimizada
    uniform_time_axis = np.linspace(start_time, end_time_coach_b, config.time_steps)

    interpolated_data = {
        'battery_id': [row['battery_id']] * config.time_steps,
        'cycle_number': [row['cycle_number']] * config.time_steps,
        'Capacity_Ah': [capacity] * config.time_steps,
        'Energy_Wh': [total_energy_wh] * config.time_steps,  # <--- NOVA LINHA
        'Ambient_Temp': [row.get('ambient_temperature', 24)] * config.time_steps,
        'SoH': [soh_value] * config.time_steps,
    }

    for feat in config.features:
        if feat.startswith('HI_'):
            interpolated_data[feat] = [hi_values.get(feat, 0.0)] * config.time_steps
        else:
            feat_arr = coach_b_df[feat].to_numpy()
            interpolated_data[feat] = np.interp(uniform_time_axis, time_arr, feat_arr)

    return pl.DataFrame(interpolated_data)


def process_nasa_dataset(config):
    logger.info("Iniciando processamento com Polars...")
    os.makedirs(config.output_dir, exist_ok=True)

    metadata_path = os.path.join(config.input_dir, "metadata.csv")
    metadata = pl.read_csv(metadata_path, ignore_errors=True).with_columns(
        pl.col('battery_id').cast(pl.Utf8)
    )

    discharge_metadata = metadata.filter(
        (pl.col('type') == 'discharge') &
        (~pl.col('battery_id').is_in(config.excluded_batteries))
    ).with_columns(
        pl.int_range(1, pl.len() + 1).over('battery_id').alias('cycle_number')
    )

    processed_dfs = []

    # Execução sequencial: o Polars e Numpy já farão isso voar
    for row in tqdm(discharge_metadata.iter_rows(named=True),
                    total=discharge_metadata.height,
                    desc="Processando Ciclos"):

        result = process_single_cycle(row, config)
        if result is not None:
            processed_dfs.append(result)

    if not processed_dfs:
        logger.error("Nenhum dado processado.")
        return None

    full_dataset_pl = pl.concat(processed_dfs)
    csv_data_path = os.path.join(config.output_dir, 'battery_health_dataset.csv')
    full_dataset_pl.write_csv(csv_data_path)
    logger.info(f"Dataset salvo em {csv_data_path}")

    return full_dataset_pl