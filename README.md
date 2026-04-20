# Predicao de State of Health (SoH) de Baterias com Aprendizado Baseado em Spikes

## Resumo
Este repositorio apresenta um pipeline experimental para estimacao de **State of Health (SoH)** em baterias de ions de litio, com base no dataset da NASA. O estudo compara arquiteturas de aprendizado profundo classico, modelos baseados em Transformer, variantes fisicamente informadas e modelos neuromorficos baseados em redes neurais de disparos (Spiking Neural Networks, SNNs).

A proposta metodologica combina preprocessamento temporal de ciclos de descarga, avaliacao com particionamento por grupos de baterias (para reduzir vazamento de informacao), validacao cruzada estratificada por dominio fisico e avaliacao holdout em conjunto inedito. O trabalho tambem contempla avaliacao fora de dominio (OOD) e quantizacao pos-treinamento para cenarios de inferencia em borda.

## Objetivos
- Estimar SoH em regime supervisionado a partir de sinais eletroquimicos medidos ao longo do tempo.
- Comparar o desempenho preditivo de familias de modelos heterogeneas sob o mesmo protocolo experimental.
- Investigar robustez de generalizacao em distribuicoes fora do dominio de treino.
- Avaliar viabilidade de compressao de modelos via quantizacao dinamica.

## Contribuicoes Metodologicas
- **Particionamento por grupos (`battery_id`)** em todas as etapas de avaliacao.
- **Estratificacao por dominio fisico de bateria** na validacao cruzada (`StratifiedGroupKFold`).
- **Pipeline unificado** para treino de modelos classicos, Transformer, fisicos e SNN.
- **Relatorios reprodutiveis** em CSV (metricas por fold e agregadas).
- **Quantizacao pos-treinamento (PTQ)** para reduzir custo de armazenamento.

## Estrutura do Repositorio
```text
config/              # Configuracoes centrais do experimento
  settings.py

data/                # Ingestao e processamento do dataset NASA
  ingestion.py

features/            # Construcao de sequencias temporais
  builder.py

models/              # Arquiteturas de rede
  classic_dl.py
  transformers.py
  physics.py
  spiking.py
  factory.py

training/            # Rotinas de treino por familia de modelo
  loops.py

evaluation/          # Protocolos de validacao e teste
  cross_val.py
  holdout.py

visualization/       # Geracao de graficos
  plots.py

output/
  models/            # Pesos por fold (cross-validation)
  holdout_models/    # Pesos finais (holdout)
  reports/           # Resultados tabulares (CSV)
  plots/             # Visualizacoes

main.py              # Execucao principal do pipeline
check_splits.py      # Auditoria de particionamentos
ood_val.py           # Avaliacao out-of-domain
search.py            # Busca de hiperparametros
quantize.py          # Quantizacao de modelos
```

## Descricao do Pipeline
1. **Ingestao e limpeza** (`data/ingestion.py`)
   - Leitura de `metadata.csv` e ciclos de descarga.
   - Exclusao de baterias definidas em configuracao.
   - Truncamento do ciclo por limiar de tensao.
   - Estimativa de capacidade por integracao de corrente.
   - Calculo de SoH e interpolacao para eixo temporal uniforme.

2. **Construcao de sequencias** (`features/builder.py`)
   - Agrupamento por (`battery_id`, `cycle_number`).
   - Formacao de tensores 3D no formato `[N, T, F]`.
   - Retorno de `X`, `y` e `groups` para avaliacao com grupos.

3. **Separacao global do holdout** (`main.py`)
   - Aplicacao de `GroupShuffleSplit` para definir cofre de teste inedito.

4. **Validacao cruzada estratificada** (`evaluation/cross_val.py`)
   - `StratifiedGroupKFold` com estratos de dominio fisico.
   - Split interno treino/validacao via `GroupShuffleSplit`.
   - Escalonamento com `StandardScaler` ajustado apenas no treino.
   - Treino e avaliacao de todas as arquiteturas por fold.

5. **Treino final e avaliacao holdout** (`evaluation/holdout.py`)
   - Treino em conjunto de desenvolvimento.
   - Validacao interna por grupos.
   - Teste final no conjunto inedito.

6. **Quantizacao e visualizacao** (`utils/common.py`, `visualization/plots.py`)
   - PTQ dinamica em camadas lineares.
   - Geracao de curvas de perda e comparativos de desempenho.

## Modelos Investigados
- **SNN (SpikingJelly):**
  - `SJ-Spiking-MultiStep`
  - `SJ-Spiking-Attention`
  - `SJ-Spiking-Hybrid`
- **Classicos:**
  - `CNN-1D`
  - `LSTM`
  - `CNN-LSTM`
- **Transformer e Hibridos:**
  - `iTransformer`
  - `DynamicGraph-iTR`
  - `Phys-iTransformer`
  - `Phys-iTR-Curriculum`

## Hiperparametros de Referencia
Arquivo: `output/reports/best_hyperparameters.json`

```json
{
  "d_model": 64,
  "num_heads": 4,
  "tau": 10.0,
  "surrogate": "SoftSign",
  "surrogate_alpha": 6.0,
  "learning_rate": 0.0006838174674743018
}
```

## Resultados Experimentais
### Cross-validation (media por modelo)
Fonte: `output/reports/cv_summary_results.csv`

- `SJ-Spiking-Hybrid`: R2 medio = 0.8410, MAE medio = 1.8089
- `SJ-Spiking-MultiStep`: R2 medio = 0.8324, MAE medio = 1.4663
- `Phys-iTR-Curriculum`: R2 medio = 0.6977, MAE medio = 2.4462
- `iTransformer`: R2 medio = 0.6161, MAE medio = 2.9798
- `LSTM`: degradacao consistente (R2 medio negativo)

Observacao: o desempenho medio sugere vantagem das abordagens SNN no protocolo de validacao cruzada adotado, com variancias distintas entre folds.

### Holdout (teste inedito)
Fonte: `output/reports/holdout_results.csv`

- `SJ-Spiking-MultiStep`: MAE = 0.4658, RMSE = 0.5358, R2 = 0.9923
- `CNN-LSTM`: MAE = 0.4662, RMSE = 0.7597, R2 = 0.9845
- `SJ-Spiking-Hybrid`: MAE = 0.6695, RMSE = 0.9312, R2 = 0.9768

No cenario holdout, observa-se alto poder preditivo para os melhores modelos, com destaque para `SJ-Spiking-MultiStep`.

### Avaliacao OOD (fora de dominio)
Fonte: `output/reports/ood_results.csv`

- `SJ-Spiking-MultiStep`: R2 = 0.0106
- `iTransformer`: R2 = -1.2198
- `CNN-LSTM`: R2 = -2.3565

A queda de desempenho em OOD indica sensibilidade a mudancas de distribuicao e motiva estudos adicionais de robustez e adaptacao de dominio.

## Como Executar
> Ajuste os parametros em `config/settings.py` conforme seu ambiente e objetivo experimental.

Executar pipeline completo:
```bash
python main.py
```

Auditar splits por bateria/dominio:
```bash
python check_splits.py
```

Executar busca de hiperparametros:
```bash
python search.py
```

Executar avaliacao OOD:
```bash
python ood_val.py
```

Executar quantizacao pos-treinamento:
```bash
python quantize.py
```

## Artefatos Gerados
- Pesos por fold: `output/models/`
- Pesos finais de holdout: `output/holdout_models/`
- Relatorios tabulares: `output/reports/`
  - `cv_summary_results.csv`
  - `cv_detailed_results.csv`
  - `holdout_results.csv`
  - `ood_results.csv`
- Figuras: `output/plots/`

## Ameaças a Validade e Limitacoes
- **Dependencia de distribuicao:** reducao expressiva em cenario OOD.
- **Variabilidade entre folds:** alguns modelos mostram alta dispersao de metricas.
- **Representatividade dos sinais:** desempenho pode depender da janela temporal escolhida.

## Trabalhos Futuros
- Estrategias de robustez OOD (domain adaptation, fine-tuning por dominio).
- Ensembles entre modelos complementares (ex.: SNN + CNN-LSTM).
- Modelagem de incerteza e calibracao das predicoes.
- Avaliacao de custo computacional/energetico em hardware de borda.

## Reprodutibilidade
Para reproducao consistente, recomenda-se:
- Fixar sementes (`set_seed`) e registrar versoes de bibliotecas.
- Preservar os mesmos arquivos de entrada e mapeamentos de dominio.
- Versionar artefatos de configuracao e relatarios experimentais.

## Licenca
Defina aqui a licenca do projeto (por exemplo, MIT ou Apache-2.0).

