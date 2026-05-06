import os
import json
import torch
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # reprodutibilidade
    seed: int = 42

    # Device (Resolvido via __post_init__ para evitar problemas de avaliação no momento da importação)
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
    device: torch.device = field(init=False)

    # paths
    input_dir: str = "./input/nasa-battery-dataset/cleaned_dataset"
    output_dir: str = "./output/processed"
    optuna_study_dir: str = "./output/optuna_study"
    models_dir: str = "./output/models"
    holdout_models_dir: str = "./output/holdout_models"
    plots_cv_dir: str = "./output/plots/cross_val"
    plots_holdout_dir: str = "./output/plots/holdout"
    test_size: float = 0.2
    val_size: float = 0.15
    reports_dir: str = "./output/reports"
    use_synthetic_data: bool = False

    # Data parameters
    observation_window_sec: int = 600
    time_steps: int = 50
    features: tuple = (
        'Voltage_measured',
        'Current_measured',
        'Temperature_measured',
        'SoC',
        'HI_Time_of_Discharge',
        'HI_Max_Temp',
        'HI_Voltage_Integral',
        'HI_Voltage_Drop',
        'HI_Temp_Delta',
        'HI_Mean_Temp',
        'HI_Thermal_Integral',
        'HI_Temp_Rate',
        'HI_Voltage_Efficiency',
        'HI_Time_to_Max_Temp',
        'HI_Final_Voltage',
        'HI_Internal_Resistance'
    )
    excluded_batteries: tuple = ('B0049', 'B0050', 'B0051', 'B0052')

    # Training Gerais
    batch_size: int = 64
    epochs: int = 150
    patience: int = 20
    weight_decay: float = 1e-4
    noise_std: float = 0.02

    # =========================================================
    # HIPERPARÂMETROS DA REDE (Defaults)
    # Estes valores serão sobrescritos se o JSON do Optuna existir
    # =========================================================
    learning_rate: float = 1e-3
    d_model: int = 64
    num_heads: int = 4
    tau: float = 2.0
    surrogate: str = 'ATan'
    surrogate_alpha: float = 2.0
    lambda_phys: float = 2.0
    # Validation
    k_folds: int = 4
    hi_correlation_threshold: float = 0.60

    # Estratificação Física
    BATTERY_DOMAINS: dict = field(default_factory=lambda: {
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
    })

    def __post_init__(self):
        """Executado automaticamente ao fazer `config = PipelineConfig()`"""
        # Inicializa o device de forma segura
        self.device = torch.device(self.device_name)

        # Caminho para o ficheiro JSON do Optuna
        best_params_path = os.path.join(self.reports_dir, 'best_hyperparameters.json')

        if self.input_dir == "./input/synthetic_smartphones":
            self.use_synthetic_data = True
            self.BATTERY_DOMAINS = {}
            for i in range(0, 10):
                self.BATTERY_DOMAINS[str(i)] = 'budget_phone'
            for i in range(10, 20):
                self.BATTERY_DOMAINS[str(i)] = 'premium_flagship'
            for i in range(20, 30):
                self.BATTERY_DOMAINS[str(i)] = 'gaming_phone'

        # Se o ficheiro existir, injeta os valores na configuração
        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, 'r') as f:
                    best_params = json.load(f)

                # Atualiza dinamicamente as variáveis da dataclass
                for key, value in best_params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                print(f"⚙️ [PipelineConfig] Hiperparâmetros otimizados carregados de: {best_params_path}")
            except Exception as e:
                print(f"⚠️ [PipelineConfig] Erro ao carregar JSON de hiperparâmetros (usando defaults): {e}")
        else:
            print("⚙️ [PipelineConfig] JSON do Optuna não encontrado. A usar hiperparâmetros por defeito.")
    def save(self, filepath: str):
        """Serializa o estado atual da config (incluindo features selecionadas) para JSON."""
        import dataclasses
        state = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.device):
                state[f.name] = str(val)
            elif isinstance(val, (tuple, list)):
                state[f.name] = list(val)
            elif isinstance(val, dict):
                state[f.name] = val
            else:
                state[f.name] = val
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as fp:
            json.dump(state, fp, indent=2)
        print(f"⚙️ [PipelineConfig] Estado salvo em: {filepath}")