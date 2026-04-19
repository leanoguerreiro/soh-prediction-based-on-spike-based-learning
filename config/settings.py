import torch
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # reprodutibilidade
    seed: int = 123
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths
    input_dir: str = "./input/nasa-battery-dataset/cleaned_dataset"
    output_dir: str = "./output/processed"
    optuna_study_dir: str = "./output/optuna_study"
    models_dir: str = "./output/models"
    holdout_models_dir: str = "./output/holdout_models"
    plots_cv_dir: str = "./output/plots/cross_val"
    plots_holdout_dir: str = "./output/plots/holdout"
    test_size: float = 0.2
    val_size: float = 0.1
    reports_dir: str = "./output/reports"

    # Data parameters
    observation_window_sec: int = 600
    time_steps: int = 50
    features: tuple = ('Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC')
    excluded_batteries: tuple = ('B0049', 'B0050', 'B0051', 'B0052')

    # Training
    batch_size: int = 64
    epochs: int = 150
    patience: int = 20
    learning_rate: float = 0.001146714835079474
    weight_decay: float = 1e-4
    noise_std: float = 0.02

    # Validation
    k_folds: int = 4

    BATTERY_DOMAINS = {
        # 24°C (Temperatura Ambiente)
        'B0005': 'Ambiente_2A', 'B0006': 'Ambiente_2A', 'B0007': 'Ambiente_2A', 'B0018': 'Ambiente_2A',
        'B0025': 'Ambiente_Onda', 'B0026': 'Ambiente_Onda', 'B0027': 'Ambiente_Onda', 'B0028': 'Ambiente_Onda',
        'B0033': 'Ambiente_4A', 'B0034': 'Ambiente_4A', 'B0036': 'Ambiente_2A',

        # 43°C / 44°C (Temperatura Quente)
        'B0029': 'Quente_4A', 'B0030': 'Quente_4A', 'B0031': 'Quente_4A', 'B0032': 'Quente_4A',
        'B0038': 'Quente_Multi', 'B0039': 'Quente_Multi', 'B0040': 'Quente_Multi',

        # 4°C (Temperatura Fria)
        'B0041': 'Frio_Multi', 'B0042': 'Frio_Multi', 'B0043': 'Frio_Multi', 'B0044': 'Frio_Multi',
        'B0045': 'Frio_1A', 'B0046': 'Frio_1A', 'B0047': 'Frio_1A', 'B0048': 'Frio_1A',
        'B0053': 'Frio_2A', 'B0054': 'Frio_2A', 'B0055': 'Frio_2A', 'B0056': 'Frio_2A'
    }