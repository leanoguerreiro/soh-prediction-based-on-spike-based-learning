import os
import json
import torch
import logging
import optuna
import numpy as np
from optuna.trial import TrialState
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# Importações do seu pipeline
from models.spiking import SJ_Spiking_MultiStep_Attention
from training.loops import train_spikingjelly
from utils.common import setup_logger, set_seed
from features.builder import build_sequences
from data.ingestion import process_nasa_dataset
from config.settings import PipelineConfig


def prepare_data_for_optuna(config):
    """Prepara os dados respeitando o Cofre (Holdout) para evitar Data Leakage."""
    logger = logging.getLogger("GridSearch")
    logger.info("Carregando e preparando dados para o Optuna...")

    df = process_nasa_dataset(config)
    X, y, groups = build_sequences(df, config)
    n_features = X.shape[2]

    # 1. SEPARAÇÃO GLOBAL (O COFRE) - Idêntico ao main.py para garantir consistência
    gss_global = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    dev_idx, holdout_idx = next(gss_global.split(X, y, groups))

    X_dev, y_dev, groups_dev = X[dev_idx], y[dev_idx], groups[dev_idx]

    # 2. Separar Treino e Validação APENAS do conjunto de Desenvolvimento
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
    train_idx, val_idx = next(gss_val.split(X_dev, y_dev, groups_dev))

    X_train, y_train = X_dev[train_idx], y_dev[train_idx]
    X_val, y_val = X_dev[val_idx], y_dev[val_idx]

    # 3. Escalonamento
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

    # 4. Conversão para Tensores
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(config.device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(config.device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(config.device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(config.device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)

    return train_loader, X_val_t, y_val_t, y_val, n_features


def objective(trial, config, train_loader, X_val_t, y_val_t, y_val, n_features):
    """Função objetivo que o Optuna tentará minimizar (MAE)."""

    # 1. Hiperparâmetros da Arquitetura
    d_model = trial.suggest_categorical('d_model', [4, 8, 16, 32, 64])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16, 32])

    # Poda (Pruning) Imediata: se a divisão não for exata, abortamos antes de dar erro
    if d_model % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    # 2. Parâmetros Físicos do SNN
    tau = trial.suggest_float('tau', 2.0, 10.0, step=1.0)

    # Função Surrogate
    surrogate_name = trial.suggest_categorical('surrogate', ['Sigmoid', 'ATan', 'SoftSign'])
    surrogate_alpha = trial.suggest_float('surrogate_alpha', 1.0, 8.0, step=0.5)

    if surrogate_name == 'ATan':
        from spikingjelly.activation_based import surrogate
        surrogate_fn = surrogate.ATan(alpha=surrogate_alpha)
    elif surrogate_name == 'SoftSign':
        from spikingjelly.activation_based import surrogate
        surrogate_fn = surrogate.SoftSign(alpha=surrogate_alpha)
    else:
        from spikingjelly.activation_based import surrogate
        surrogate_fn = surrogate.Sigmoid(alpha=surrogate_alpha)

    # Learning Rate
    config.learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)

    # 3. Instanciação do Modelo
    model = SJ_Spiking_MultiStep_Attention(
        time_steps=config.time_steps,
        n_features=n_features,
        d_model=d_model,
        num_heads=num_heads,
        tau=tau,
        surrogate_fn=surrogate_fn
    ).to(config.device)

    # 4. Treinamento
    try:
        preds, _ = train_spikingjelly(model, train_loader, X_val_t, y_val_t, X_val_t, config)
        mae = mean_absolute_error(y_val, preds)
    except Exception as e:
        # Se a rede explodir (NaNs) ou der erro de CUDA por hiperparâmetro extremo,
        # o Optuna simplesmente descarta a tentativa em vez de crashar o código.
        raise optuna.exceptions.TrialPruned()

    return mae


def run_optuna_search(config, n_trials=50):
    logger = setup_logger("GridSearch")
    logger.info("=== INICIANDO OPTUNA BAYESIAN SEARCH NO SJ-SPIKING-MULTISTEP ===")

    # Criar diretórios necessários
    os.makedirs(config.optuna_study_dir, exist_ok=True)
    reports_dir = getattr(config, 'reports_dir', './output/reports')
    os.makedirs(reports_dir, exist_ok=True)

    # Carrega os dados apenas 1x (RESPEITANDO O COFRE)
    train_loader, X_val_t, y_val_t, y_val, n_features = prepare_data_for_optuna(config)

    # Cria o "Study" do Optuna (queremos MINIMIZAR o MAE)
    study = optuna.create_study(
        study_name="snn_soh_optimization",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed)
    )

    # Função lambda para injetar os dados pré-carregados no objective
    func = lambda trial: objective(trial, config, train_loader, X_val_t, y_val_t, y_val, n_features)

    # Executa as tentativas
    logger.info(f"Iniciando busca de hiperparâmetros ({n_trials} trials permitidos)...")
    study.optimize(func, n_trials=n_trials)

    # Resultados Finais
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    logger.info("\n=== ESTATÍSTICAS DA BUSCA (OPTUNA) ===")
    logger.info(f"  Trials Completos: {len(complete_trials)}")
    logger.info(f"  Trials Descartados (Pruned/Erros): {len(pruned_trials)}")

    logger.info("\n🏆 MELHOR MODELO ENCONTRADO:")
    trial = study.best_trial
    logger.info(f"  MAE de Validação: {trial.value:.4f}")

    # ==========================================================
    # SALVAMENTO EM JSON (PARA NUNCA MAIS PERDER OS RESULTADOS)
    # ==========================================================
    best_params_path = os.path.join(reports_dir, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)

    logger.info("  Hiperparâmetros:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Salva o histórico completo num CSV
    df_results = study.trials_dataframe()
    csv_path = os.path.join(reports_dir, 'optuna_full_history.csv')
    df_results.to_csv(csv_path, index=False)

    logger.info(f"\n💾 Melhor hiperparâmetro salvo em: {best_params_path}")
    logger.info(f"💾 Histórico completo salvo em: {csv_path}")


if __name__ == "__main__":
    config = PipelineConfig()
    set_seed(config.seed)

    # Opcional: Aumentar n_trials para 100 se for rodar de madrugada
    run_optuna_search(config, n_trials=200)
