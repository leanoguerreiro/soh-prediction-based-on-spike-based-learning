import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# Importações do seu projeto
from config.settings import PipelineConfig
from utils.common import set_seed
from data.ingestion import process_nasa_dataset
from features.builder import build_sequences


def format_bats(baterias, config):
    """Função auxiliar para formatar a lista de baterias com os seus domínios físicos"""
    formatted = []
    for b in baterias:
        dominio = config.BATTERY_DOMAINS.get(b, 'Desconhecido')
        formatted.append(f"{b} [{dominio}]")
    return "\n      ".join(formatted)


def main():
    config = PipelineConfig()
    set_seed(config.seed)

    print("\n" + "=" * 80)
    print(f"🔋 MAPEAMENTO ESTRATIFICADO DE BATERIAS NO PIPELINE (SEED: {config.seed}) 🔋")
    print("=" * 80)

    # 1. Carregar dados e extrair os grupos (IDs das baterias)
    df = process_nasa_dataset(config)
    X, y, groups = build_sequences(df, config)

    todas_baterias = np.unique(groups)
    print(f"\nTotal de Baterias Únicas Processadas: {len(todas_baterias)}")

    # ==========================================================================
    # 2. O COFRE (SEPARAÇÃO GLOBAL)
    # ==========================================================================
    gss_global = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    dev_idx, holdout_idx = next(gss_global.split(X, y, groups))

    groups_dev = groups[dev_idx]
    groups_holdout = groups[holdout_idx]

    baterias_dev = np.unique(groups_dev)
    baterias_holdout = np.unique(groups_holdout)

    print("\n" + "=" * 80)
    print("📌 HOLDOUT EVALUATION (TREINO FINAL)")
    print("=" * 80)
    print(f"🔒 TESTE INÉDITO (Cofre) -> {len(baterias_holdout)} baterias:")
    print(f"      {format_bats(baterias_holdout, config)}\n")

    # Separação interna do Treino Final (Treino vs Validação)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.seed)
    train_idx, val_idx = next(gss_val.split(X[dev_idx], y[dev_idx], groups_dev))

    baterias_holdout_train = np.unique(groups_dev[train_idx])
    baterias_holdout_val = np.unique(groups_dev[val_idx])

    print(f"⚙️ TREINO FINAL -> {len(baterias_holdout_train)} baterias:")
    print(f"      {format_bats(baterias_holdout_train, config)}\n")
    print(f"📊 VALIDAÇÃO FINAL -> {len(baterias_holdout_val)} baterias:")
    print(f"      {format_bats(baterias_holdout_val, config)}\n")

    # ==========================================================================
    # 3. CROSS-VALIDATION ESTRATIFICADO (APENAS NOS DADOS DEV)
    # ==========================================================================
    print("=" * 80)
    print(f"🔄 CROSS-VALIDATION ({config.k_folds}-FOLD STRATIFIED NOS DADOS DE DESENVOLVIMENTO)")
    print("=" * 80)

    # NOVO: Usar StratifiedGroupKFold com o mapeamento físico
    sgkf = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    # Criar o array de estratificação (y_stratum) para os dados de desenvolvimento
    y_stratum_dev = np.array([config.BATTERY_DOMAINS.get(bid, 'Desconhecido') for bid in groups_dev])

    for fold, (cv_train_val_idx, cv_test_idx) in enumerate(sgkf.split(X[dev_idx], y_stratum_dev, groups_dev)):
        print(f"\n--- FOLD {fold + 1} ---")

        # Teste do Fold atual
        fold_test_groups = groups_dev[cv_test_idx]
        baterias_fold_test = np.unique(fold_test_groups)

        # Treino e Val do Fold atual
        fold_tv_groups = groups_dev[cv_train_val_idx]

        # O Cross-Val faz um split interno para Validação (usando test_size=0.2 fixo)
        gss_cv_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        dummy_X = np.zeros((len(fold_tv_groups), 1))
        cv_t_idx, cv_v_idx = next(gss_cv_val.split(dummy_X, dummy_X, fold_tv_groups))

        baterias_fold_train = np.unique(fold_tv_groups[cv_t_idx])
        baterias_fold_val = np.unique(fold_tv_groups[cv_v_idx])

        print(f"   ► Teste ({len(baterias_fold_test)} baterias):")
        print(f"      {format_bats(baterias_fold_test, config)}\n")
        print(f"   ► Validação ({len(baterias_fold_val)} baterias):")
        print(f"      {format_bats(baterias_fold_val, config)}\n")
        # O treino tem muitas baterias, podemos imprimir de forma mais compacta se preferir, mas formatado fica excelente para auditoria
        print(f"   ► Treino ({len(baterias_fold_train)} baterias):")
        print(f"      {format_bats(baterias_fold_train, config)}")


if __name__ == "__main__":
    main()