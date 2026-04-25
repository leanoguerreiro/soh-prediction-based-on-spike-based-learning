import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm

logger = logging.getLogger("BatteryPipeline")


def process_synthetic_dataset(config):
    logger.info("Iniciando ingestão do Dataset Sintético de Smartphones...")
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Carregar os dados gerados pelo seu script
    df_raw = pd.read_csv(os.path.join(config.input_dir, 'smartphone_battery_dataset.csv'))
    df_sum = pd.read_csv(os.path.join(config.input_dir, 'battery_cycle_summary.csv'))

    # 2. Filtrar apenas ciclos de descarga (como na NASA)
    df_raw = df_raw[df_raw['cycle_type'] == 'discharge'].copy()

    # 3. Renomear colunas brutas para o padrão do Framework
    df_raw.rename(columns={
        'voltage_V': 'Voltage_measured',
        'current_mA': 'Current_measured',
        'temp_C': 'Temperature_measured',
        'time_in_cycle': 'Time',
        'cycle_id': 'cycle_number',
        'soc': 'SoC'
    }, inplace=True)

    # 4. Renomear HIs (Health Indicators) do sumário para o padrão do Framework
    # Mapeamos 'discharge_energy_Wh' para 'HI_Voltage_Integral' por equivalência física
    df_sum.rename(columns={
        'discharge_duration_s': 'HI_Time_of_Discharge',
        'discharge_temp_max_C': 'HI_Max_Temp',
        'voltage_drop_V': 'HI_Voltage_Drop',
        'discharge_energy_Wh': 'HI_Voltage_Integral',
        'cycle_id': 'cycle_number'
    }, inplace=True)

    processed_dfs = []
    grouped = df_raw.groupby(['battery_id', 'cycle_number'])

    # 5. Interpolação e alinhamento de tensores
    for (bid, cyc), group in tqdm(grouped, desc="Interpolando Ciclos Sintéticos"):
        sum_row = df_sum[(df_sum['battery_id'] == bid) & (df_sum['cycle_number'] == cyc)]
        if sum_row.empty:
            continue

        time_arr = group['Time'].values
        if len(time_arr) < 2:
            continue

        # Cria eixo de tempo uniforme (ex: 50 passos)
        uniform_time_axis = np.linspace(time_arr[0], time_arr[-1], config.time_steps)

        # Garante que battery_id é string (para o StratifiedKFold não se perder)
        interpolated_cycle = {
            'battery_id': [str(bid)] * config.time_steps,
            'cycle_number': [cyc] * config.time_steps,
            'SoH': [group['soh'].iloc[0]] * config.time_steps
        }

        # Interpolação das séries temporais físicas
        for feat in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC']:
            f_interp = interp1d(time_arr, group[feat], kind='linear', fill_value='extrapolate')
            interpolated_cycle[feat] = f_interp(uniform_time_axis)

        # Repetição das Features Estáticas (HIs) pelo tempo da série
        for feat in ['HI_Time_of_Discharge', 'HI_Max_Temp', 'HI_Voltage_Integral', 'HI_Voltage_Drop']:
            val = sum_row[feat].iloc[0]
            interpolated_cycle[feat] = [val] * config.time_steps

        processed_dfs.append(pd.DataFrame(interpolated_cycle))

    full_dataset = pd.concat(processed_dfs)
    logger.info(f"Dataset processado com sucesso! Shape final: {full_dataset.shape}")

    return full_dataset