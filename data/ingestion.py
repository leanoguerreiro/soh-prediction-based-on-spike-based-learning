import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import logging

logger = logging.getLogger("BatteryPipeline")


def process_nasa_dataset(config):
    logger.info("Iniciando ingestão e processamento do Dataset NASA...")
    os.makedirs(config.output_dir, exist_ok=True)

    metadata_path = os.path.join(config.input_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    metadata['battery_id'] = metadata['battery_id'].astype(str)

    discharge_metadata = metadata[
        (metadata['type'] == 'discharge') &
        (~metadata['battery_id'].isin(config.excluded_batteries))
        ].copy()
    discharge_metadata['cycle_number'] = discharge_metadata.groupby('battery_id').cumcount() + 1

    processed_dfs = []

    for _, row in tqdm(discharge_metadata.iterrows(), total=len(discharge_metadata), desc="Processando Ciclos"):
        file_path = os.path.join(config.input_dir, "data", row['filename'])
        if not os.path.exists(file_path): continue

        df = pd.read_csv(file_path).copy()
        cutoff_idx = df[df['Voltage_measured'] < 2.7].index.min()
        truncated_df = df if pd.isna(cutoff_idx) else df.iloc[:cutoff_idx].copy()

        truncated_df['Time_diff_hr'] = truncated_df['Time'].diff().fillna(0) / 3600
        truncated_df['Delta_Q'] = truncated_df['Current_measured'] * truncated_df['Time_diff_hr']
        capacity = abs(truncated_df['Delta_Q'].sum())

        if capacity > 1.4:
            truncated_df['SoC'] = 100 * (1 + truncated_df['Delta_Q'].cumsum() / capacity)
            soh_value = (capacity / 2.0) * 100

            start_time = truncated_df['Time'].iloc[0]
            end_time_coach_b = start_time + config.observation_window_sec
            coach_b_df = truncated_df[truncated_df['Time'] <= end_time_coach_b].copy()

            if coach_b_df.empty or len(coach_b_df) < 5:
                continue

            uniform_time_axis = np.linspace(start_time, end_time_coach_b, config.time_steps)
            interpolated_cycle = {
                'battery_id': [row['battery_id']] * config.time_steps,
                'cycle_number': [row['cycle_number']] * config.time_steps,
                'SoH': [soh_value] * config.time_steps,
            }

            for feat in config.features:
                f_interp = interp1d(coach_b_df['Time'], coach_b_df[feat], kind='linear', fill_value='extrapolate')
                interpolated_cycle[feat] = f_interp(uniform_time_axis)

            processed_dfs.append(pd.DataFrame(interpolated_cycle))

    full_dataset = pd.concat(processed_dfs)
    csv_data_path = os.path.join(config.output_dir, 'battery_health_dataset.csv')
    full_dataset.to_csv(csv_data_path, index=False)
    logger.info(f"Dataset salvo em {csv_data_path}")

    return full_dataset