import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configurações estéticas para artigos científicos
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")
CORES_MODELOS = {
    # Família SJ-Spiking — tons de azul/ciano
    "SJ-Spiking-MultiStep": "#1f77b4",  # Azul principal
    "SJ-Spiking-Attention": "#4fc3f7",  # Azul claro
    "SJ-Spiking-Hybrid": "#0077b6",  # Azul escuro
    "SJ-Spiking-Simple": "#90e0ef",  # Azul bem claro
    'SJ-LSM': "#00b4d8",  # Ciano
    'SJ-Spiking-Dilated': "#48cae4",  # Azul vibrante

    # Família CNN/LSTM — tons de laranja/vermelho
    "CNN-1D": "#e85d04",  # Laranja escuro
    "LSTM": "#f48c06",  # Âmbar
    "CNN-LSTM": "#dc2f02",  # Vermelho-laranja
    "CNN-DILATED": "#ffba08",  # Amarelo vibrante

    # Família iTransformer — tons de roxo/rosa
    "iTransformer": "#7b2d8b",  # Roxo
    "DynamicGraph-iTR": "#c77dff",  # Lilás
    "Phys-iTR-Curriculum": "#e040fb",  # Rosa vibrante
    "Phys-iTransformer": "#f48fb1",  # Rosa claro

    # Referência
    "SoH Real": "#212121",  # Preto suave
}


def plot_scatter_real_vs_pred(y_test, holdout_preds, save_dir):
    """
    Gráfico de Dispersão (Real vs Previsto) - Foco nos 4 principais modelos.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    models_to_plot = [
        "SJ-Spiking-MultiStep",
        "SJ-Spiking-Attention",
        "SJ-Spiking-Hybrid",
        "SJ-Spiking-Simple",
        "SJ-LSM",
        "SJ-Spiking-Dilated",
        "CNN-1D",
        "LSTM",
        "CNN-LSTM",
        "CNN-DILATED",
        "iTransformer",
        "DynamicGraph-iTR",
        "Phys-iTR-Curriculum",
        "Phys-iTransformer"

    ]
    models_to_plot = [m for m in models_to_plot if m in holdout_preds]

    # Gráfico combinado (grid dinâmico)
    n = len(models_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(6 * ncols, 5 * nrows))
    for i, model_name in enumerate(models_to_plot, 1):
        plt.subplot(nrows, ncols, i)
        preds = holdout_preds[model_name]

        min_val = min(np.min(y_test), np.min(preds)) - 2
        max_val = max(np.max(y_test), np.max(preds)) + 2

        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal', alpha=0.7)
        plt.scatter(y_test, preds, alpha=0.6, edgecolors='w', s=60, color='#1f77b4')

        plt.title(f"{model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("SoH Real (%)")
        plt.ylabel("SoH Previsto (%)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "holdout_scatter_real_vs_pred.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico individual por modelo
    for model_name in models_to_plot:
        preds = holdout_preds[model_name]

        min_val = min(np.min(y_test), np.min(preds)) - 2
        max_val = max(np.max(y_test), np.max(preds)) + 2

        plt.figure(figsize=(6, 6))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal', alpha=0.7)
        plt.scatter(y_test, preds, alpha=0.6, edgecolors='w', s=60, color='#1f77b4')

        plt.title(f"{model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("SoH Real (%)")
        plt.ylabel("SoH Previsto (%)")
        plt.legend()
        plt.tight_layout()

        safe_name = model_name.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(save_dir, f"holdout_scatter_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"📊 Scatter salvo em {save_dir}")


def plot_residual_distribution(y_test, holdout_preds, save_dir):
    """
    Distribuição dos Erros Residuais por modelo (violin plot).
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    error_data = []
    for model_name, preds in holdout_preds.items():
        residuals = preds - y_test
        for res in residuals:
            error_data.append({'Model': model_name, 'Erro (Previsto - Real)': res})

    df_errors = pd.DataFrame(error_data)

    # Gráfico combinado
    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df_errors, x='Model', y='Erro (Previsto - Real)', hue='Model', inner='quartile',
                   palette='muted', legend=False)
    plt.axhline(0, color='r', linestyle='--', linewidth=1.5)
    plt.title("Distribuição do Erro Residual no Holdout", fontsize=16, fontweight='bold')
    plt.ylabel("Erro Absoluto (SoH %)")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "holdout_residual_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico individual por modelo
    for model_name in holdout_preds.keys():
        df_model = df_errors[df_errors['Model'] == model_name]

        plt.figure(figsize=(6, 6))
        df_model = df_model.copy()
        df_model['_grp'] = model_name
        sns.violinplot(data=df_model, y='Erro (Previsto - Real)', hue='_grp',
                       inner='quartile', palette='muted', legend=False)
        plt.axhline(0, color='r', linestyle='--', linewidth=1.5)
        plt.title(f"Erro Residual — {model_name}", fontsize=14, fontweight='bold')
        plt.ylabel("Erro (Previsto - Real) (SoH %)")
        plt.tight_layout()

        safe_name = model_name.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(save_dir, f"holdout_residual_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"📊 Violin plot salvo em {save_dir}")


def generate_holdout_plots(y_test, holdout_preds, save_dir):
    """
    Gera todos os gráficos de avaliação do Holdout.
    """
    plot_scatter_real_vs_pred(y_test, holdout_preds, save_dir)
    plot_residual_distribution(y_test, holdout_preds, save_dir)
    print(f"📈 Todos os gráficos do Holdout salvos em {save_dir}")


def plot_loss_curves(histories_per_fold, fold=None, save_dir="./output/plots"):
    """Gera as curvas de treino e validação (uma imagem por modelo).

    Aceita dois formatos de `histories_per_fold`:
      - CV:      {fold_num: {model_name: history_dict}, ...}
      - Holdout: {model_name: history_dict}   (fold=None ou fold=1)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Detecta se o dict é {fold -> {name -> hist}} ou {name -> hist} diretamente
    first_val = next(iter(histories_per_fold.values()))
    is_nested = isinstance(first_val, dict) and any(
        isinstance(v, dict) and 'train_loss' in v for v in first_val.values()
    )

    if is_nested:
        # Formato CV padrão
        folds_to_plot = [fold] if fold is not None else list(histories_per_fold.keys())
        iter_items = [(f, histories_per_fold[f]) for f in folds_to_plot]
    else:
        # Formato holdout: {name: hist} — envolve num fold fictício
        fold_label = fold if fold is not None else 1
        iter_items = [(fold_label, histories_per_fold)]

    for f, histories in iter_items:
        for name, hist in histories.items():
            plt.figure(figsize=(8, 5))
            epochs = range(1, len(hist['train_loss']) + 1)

            plt.plot(epochs, hist['train_loss'], label='Train Loss', color=CORES_MODELOS.get(name, 'blue'),
                     linestyle='-')
            if 'val_loss' in hist:
                plt.plot(epochs, hist['val_loss'], label='Val Loss', color=CORES_MODELOS.get(name, 'blue'),
                         linestyle='--')

            fold_str = f"Fold {f}"
            plt.title(f"Curva de Loss: {name} - {fold_str}", fontweight='bold')
            plt.xlabel("Épocas")
            plt.ylabel("Huber Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            safe_name = name.replace("/", "_").replace(" ", "_")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"loss_{safe_name}_fold{f}.png"), dpi=300, bbox_inches='tight')
            plt.close()


def generate_all_plots(cv_results, fold_data, save_dir="./output/plots"):
    """Função mestre que orquestra a geração de todas as figuras isoladas."""
    os.makedirs(save_dir, exist_ok=True)
    print("\nGerando pacote de visualizações individuais (sem subplots)...")

    # 1. Preparar DataFrame de métricas para KDE e Boxplots
    df_metrics = _build_metrics_df(cv_results)

    # 2. Gerar Gráficos Independentes do Tempo
    _plot_metricas_medias(cv_results, save_dir)  # 01a, 01b, 01c
    _plot_radar_chart(cv_results, save_dir)  # 02
    _plot_kde_histograms(df_metrics, save_dir)  # 03a, 03b, 03c
    _plot_boxplot_mae(df_metrics, save_dir)  # 04

    # 3. Gerar Gráficos Dependentes do Tempo para TODOS OS FOLDS
    for fold_num, data in fold_data.items():
        if not isinstance(fold_num, int):
            continue

        y_true = data['y_test']
        preds_dict = data['preds']

        _plot_ced(y_true, preds_dict, save_dir, fold_num)  # 05
        _plot_heatmap_erros(y_true, preds_dict, save_dir, fold_num)  # 06
        _plot_analise_temporal(y_true, preds_dict, save_dir, fold_num)  # 07a, 07b, 07c
        _plot_foco_snn(y_true, preds_dict, save_dir, fold_num)  # 08a, 08b, 08c

    print(f"Visualizações individuais salvas com sucesso em: {save_dir}")


# ==============================================================================
# FUNÇÕES INTERNAS DE PLOTAGEM (UM GRÁFICO POR IMAGEM)
# ==============================================================================

def _build_metrics_df(cv_results):
    data_rows = []
    for model_name, metrics in cv_results.items():
        k_folds = len(metrics['mae'])
        for fold_idx in range(k_folds):
            data_rows.append({
                'Modelo': model_name,
                'Fold': fold_idx + 1,
                'MAE': metrics['mae'][fold_idx],
                'RMSE': metrics['rmse'][fold_idx],
                'R2': metrics['r2'][fold_idx]
            })
    return pd.DataFrame(data_rows)


def _plot_metricas_medias(cv_results, save_dir):
    """Gera três imagens isoladas para MAE, RMSE e R2."""
    models = list(cv_results.keys())
    paleta = [CORES_MODELOS.get(m, 'gray') for m in models]

    mae_means = [np.mean(cv_results[m]['mae']) for m in models]
    rmse_means = [np.mean(cv_results[m]['rmse']) for m in models]
    r2_means = [np.mean(cv_results[m]['r2']) for m in models]

    df_mae = pd.DataFrame({'Modelo': models, 'MAE': mae_means})
    df_rmse = pd.DataFrame({'Modelo': models, 'RMSE': rmse_means})
    df_r2 = pd.DataFrame({'Modelo': models, 'R2': r2_means})

    # 01a - MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_mae, x='Modelo', y='MAE', hue='Modelo', palette=paleta, legend=False)
    plt.title('Comparação de MAE Médio (Menor é Melhor)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01a_media_mae.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 01b - RMSE
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_rmse, x='Modelo', y='RMSE', hue='Modelo', palette=paleta, legend=False)
    plt.title('Comparação de RMSE Médio (Menor é Melhor)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01b_media_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 01c - R2
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_r2, x='Modelo', y='R2', hue='Modelo', palette=paleta, legend=False)
    plt.title('Comparação de R² Médio (Maior é Melhor)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01c_media_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_radar_chart(cv_results, save_dir):
    """02: Gráfico de Radar (Mantido como um único gráfico)."""
    models = list(cv_results.keys())
    categories = ['MAE (Inv)', 'RMSE (Inv)', 'R² Score']
    N = len(categories)

    mae_means = np.array([np.mean(cv_results[m]['mae']) for m in models])
    rmse_means = np.array([np.mean(cv_results[m]['rmse']) for m in models])
    r2_means = np.array([np.mean(cv_results[m]['r2']) for m in models])

    mae_norm = 1 - (mae_means - mae_means.min()) / (mae_means.max() - mae_means.min() + 1e-6)
    rmse_norm = 1 - (rmse_means - rmse_means.min()) / (rmse_means.max() - rmse_means.min() + 1e-6)
    r2_norm = np.clip((r2_means - r2_means.min()) / (r2_means.max() - r2_means.min() + 1e-6), 0, 1)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)

    for i, model in enumerate(models):
        values = [mae_norm[i], rmse_norm[i], r2_norm[i]]
        values += values[:1]
        color = CORES_MODELOS.get(model, 'gray')
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)

    plt.title('Radar Chart Global (Quanto mais exterior, melhor)', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))
    plt.savefig(os.path.join(save_dir, '02_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_kde_histograms(df_metrics, save_dir):
    """Gera três imagens isoladas para histogramas MAE, RMSE e R2."""
    paleta = [CORES_MODELOS.get(m, 'gray') for m in df_metrics['Modelo'].unique()]

    # 03a - Histograma MAE
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_metrics, x='MAE', hue='Modelo', multiple='layer', kde=True, bins=15, alpha=0.4, palette=paleta)
    plt.title('Distribuição MAE por Fold (KDE)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03a_hist_mae.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 03b - Histograma RMSE
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_metrics, x='RMSE', hue='Modelo', multiple='layer', kde=True, bins=15, alpha=0.4,
                 palette=paleta)
    plt.title('Distribuição RMSE por Fold (KDE)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03b_hist_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 03c - Histograma R2
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_metrics, x='R2', hue='Modelo', multiple='layer', kde=True, bins=15, alpha=0.4, palette=paleta)
    plt.title('Distribuição R² por Fold (KDE)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03c_hist_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_boxplot_mae(df_metrics, save_dir):
    """04_boxplot_mae: Variância do MAE."""
    plt.figure(figsize=(10, 8))
    paleta = [CORES_MODELOS.get(m, 'gray') for m in df_metrics['Modelo'].unique()]

    sns.boxplot(data=df_metrics, x='MAE', y='Modelo', hue='Modelo', palette=paleta, legend=False, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8})
    sns.stripplot(data=df_metrics, x='MAE', y='Modelo', color='black', alpha=0.5, size=6)

    plt.title("Estabilidade do Erro MAE (Variação entre os Folds)", fontweight='bold')
    plt.xlabel("MAE (%)")
    plt.ylabel("Modelo")
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_boxplot_mae.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_ced(y_true, preds_dict, save_dir, fold_num):
    """05_ced: Cumulative Error Distribution."""
    plt.figure(figsize=(10, 6))
    for name, preds in preds_dict.items():
        errors = np.abs(y_true - preds)
        sorted_errors = np.sort(errors)
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative_prob, label=name, color=CORES_MODELOS.get(name, 'gray'), linewidth=2)

    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    plt.title(f"Cumulative Error Distribution (CED) - Fold {fold_num}", fontweight='bold')
    plt.xlabel("Erro Absoluto (SoH %)")
    plt.ylabel("Proporção de Amostras")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'05_ced_fold{fold_num}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_heatmap_erros(y_true, preds_dict, save_dir, fold_num):
    """06_heatmap_erros: Mapa de calor do erro ao longo do tempo."""
    models = list(preds_dict.keys())
    error_matrix = np.zeros((len(models), len(y_true)))

    for i, name in enumerate(models):
        error_matrix[i, :] = np.abs(y_true - preds_dict[name])

    plt.figure(figsize=(14, 6))
    sns.heatmap(error_matrix, cmap="coolwarm", yticklabels=models, xticklabels=False)
    plt.title(f"Heatmap de Erros Temporais - Fold {fold_num}", fontweight='bold')
    plt.xlabel("Amostras Temporais")
    plt.ylabel("Modelos")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'06_heatmap_erros_fold{fold_num}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_analise_temporal(y_true, preds_dict, save_dir, fold_num, prefix="07", models_to_plot=None):
    """Desmembra Tendência, Scatter e Erro Móvel em 3 imagens isoladas."""
    if models_to_plot is None:
        models_to_plot = ["SJ-Spiking-MultiStep", "SJ-Spiking-Attention", "CNN-LSTM", "iTransformer"]

    models_to_plot = [m for m in models_to_plot if m in preds_dict]
    if not models_to_plot: return

    # A. Curva de Tendência
    plt.figure(figsize=(12, 6))
    limite = min(800, len(y_true))
    plt.plot(y_true[:limite], label="SoH Real", linewidth=3, color="black")
    for name in models_to_plot:
        plt.plot(preds_dict[name][:limite], label=name, color=CORES_MODELOS.get(name, 'gray'), alpha=0.8, linewidth=2)
    plt.title(f"Tendência de SoH - Fold {fold_num}", fontweight='bold')
    plt.xlabel("Índice de Teste")
    plt.ylabel("SoH (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}a_tendencia_fold{fold_num}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # B. Scatter Plot
    plt.figure(figsize=(8, 8))
    min_v = min(y_true.min(), min([preds_dict[n].min() for n in models_to_plot]))
    max_v = max(y_true.max(), max([preds_dict[n].max() for n in models_to_plot]))
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', label="Ideal (y=x)", alpha=0.5)
    for name in models_to_plot:
        plt.scatter(y_true, preds_dict[name], alpha=0.6, label=name, color=CORES_MODELOS.get(name, 'gray'), s=30)
    plt.title(f"Real vs Previsto (Scatter) - Fold {fold_num}", fontweight='bold')
    plt.xlabel("SoH Real (%)")
    plt.ylabel("SoH Previsto (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}b_scatter_fold{fold_num}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # C. Erro Móvel
    window = 15
    plt.figure(figsize=(12, 6))
    for name in models_to_plot:
        err = np.abs(y_true - preds_dict[name])
        rolling = pd.Series(err).rolling(window=window).mean()
        plt.plot(rolling, label=name, color=CORES_MODELOS.get(name, 'gray'), linewidth=2)
    plt.title(f"Erro Móvel Absoluto (Janela={window}) - Fold {fold_num}", fontweight='bold')
    plt.xlabel("Índice de Teste")
    plt.ylabel("|MAE| Móvel")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}c_erromovel_fold{fold_num}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_foco_snn(y_true, preds_dict, save_dir, fold_num):
    """08: Foco nas SNNs desmembrado em 3 imagens."""
    snns = [m for m in preds_dict.keys() if m.startswith("SJ-")]
    if not snns: return
    _plot_analise_temporal(y_true, preds_dict, save_dir, fold_num, prefix="08", models_to_plot=snns)
