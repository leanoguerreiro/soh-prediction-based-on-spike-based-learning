#!/bin/bash

# Configura o script para parar imediatamente se houver um erro
set -e

# Caminho para o Python do seu ambiente virtual
PYTHON_BIN="/home/leano/Documentos/MACHINE_LEARNING/soh/.venv/bin/python"

echo "========================================================================"
echo "🚀 INICIANDO PIPELINE COMPLETO DE BATERIAS (SOH)"
echo "========================================================================"

# 1. Busca de Hiperparâmetros (Optuna)
# Esta etapa gera o best_hyperparameters.json necessário para as próximas fases.
echo "🔍 ETAPA 1/3: Iniciando Busca Bayesiana (Optuna)..."
$PYTHON_BIN search.py

echo -e "\n✅ Hiperparâmetros otimizados e salvos com sucesso.\n"

# 2. Pipeline Principal (Cross-Val, Holdout, Quantização e Plots)
# O main.py carregará automaticamente os hiperparâmetros do JSON via PipelineConfig.
echo "📊 ETAPA 2/3: Iniciando Pipeline Principal (CV & Holdout)..."
$PYTHON_BIN main.py

echo -e "\n✅ Avaliação principal concluída e relatórios gerados.\n"

# 3. Experimento Out-of-Distribution (OOD)
# Teste de fogo final para provar a robustez física no frio (4°C).
echo "❄️ ETAPA 3/3: Iniciando Experimento Zero-Shot OOD (Frio)..."
$PYTHON_BIN ood_val.py

echo "========================================================================"
echo "🎉 PIPELINE EXECUTADO COM SUCESSO!"
echo "📂 Verifique os resultados em: ./output/reports e ./output/plots"
echo "========================================================================"