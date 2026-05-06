# Predição de State of Health (SoH) de Baterias com Modelos Profundos e Neuromórficos

## Visão geral

Este repositório implementa um pipeline experimental para estimar o **State of Health (SoH)** de baterias de íons de
lítio a partir de ciclos de descarga do dataset da NASA. O projeto compara, sob um protocolo único de pré-processamento
e avaliação, quatro famílias de modelos:

- redes neurais profundas clássicas;
- Transformers e variantes híbridas;
- modelos fisicamente informados;
- redes neurais de disparos (**Spiking Neural Networks**, SNNs) com `SpikingJelly`.

O fluxo atual do projeto cobre:

- ingestão e limpeza do dataset real da NASA;
- versão sintética opcional baseada em smartphone batteries;
- construção de sequências temporais normalizadas;
- seleção automática de variáveis por correlação no conjunto de desenvolvimento;
- validação cruzada com particionamento por grupo e por domínio físico;
- holdout global com conjunto de teste inédito;
- avaliação fora de domínio (OOD) no subconjunto frio;
- quantização pós-treinamento focada no readout dos modelos SNN;
- geração de relatórios e visualizações.

---

## Arquitetura e fluxo do sistema

O pipeline modular é organizado em camadas:

1. **Configuração central** em `config/settings.py`
    - Define caminhos, hiperparâmetros padrão, device, sementes, janelas temporais e o mapeamento
      `battery_id -> domínio físico`.
    - Carrega automaticamente `output/reports/best_hyperparameters.json` quando ele existe.

2. **Ingestão de dados** em `data/`
    - `ingestion.py` processa o dataset real da NASA.

3. **Construção e seleção de features** em `features/`
    - `builder.py` monta tensores tridimensionais `[amostras, tempo, features]`.
    - `selection.py` calcula correlação de Spearman entre features e SoH e remove variáveis pouco informativas,
      preservando sensores físicos base.

4. **Modelos** em `models/`
    - Implementações PyTorch para CNNs, LSTM, híbridos CNN-LSTM, Transformers, variantes físicas e SNNs.
    - `factory.py` instancia todas as arquiteturas com base no `PipelineConfig`.

5. **Treinamento** em `training/loops.py`
    - Loops específicos para modelos padrão, físicos, curriculum learning e SNNs.
    - Early stopping com `HuberLoss`, `AdamW`, ruído gaussiano de regularização e reset explícito dos estados de rede
      spiking.

6. **Avaliação** em `evaluation/`
    - `cross_val.py` executa `StratifiedGroupKFold` com estratificação por domínio físico.
    - `holdout.py` executa o treino final e testa no cofre de holdout.

7. **Visualização** em `visualization/plots.py`
    - Gera curvas de loss, gráficos de comparação por fold, CED, scatter real vs. previsto, histogramas, radar chart e
      análises temporais.

8. **Scripts auxiliares** na raiz
    - `main.py`, `search.py`, `ood_val.py`, `quantize.py`, `check_splits.py`, `run_pipeline.sh`.

---

## Estrutura de diretórios

```text
soh/
├── config/
│   └── settings.py                # Configuração central do pipeline
├── data/
│   ├── ingestion.py               # Ingestão do dataset NASA
├── features/
│   ├── builder.py                 # Montagem de sequências [N, T, F]
│   └── selection.py               # Seleção de features por correlação
├── models/
│   ├── classic_dl.py              # CNN1D, LSTM, CNN-LSTM, Dilated CNN
│   ├── transformers.py            # iTransformer e DynamicGraph-iTR
│   ├── physics.py                 # Modelos fisicamente informados
│   ├── spiking.py                 # SNNs e camadas neuromórficas
│   └── factory.py                 # Fábrica de modelos
├── training/
│   └── loops.py                   # Loops de treino e inferência
├── evaluation/
│   ├── cross_val.py               # Cross-validation estratificada por grupo
│   └── holdout.py                 # Avaliação final holdout
├── visualization/
│   └── plots.py                   # Geração de figuras e relatórios visuais
├── output/
│   ├── processed/                 # Dataset processado
│   ├── models/                    # Checkpoints e scalers da cross-validation
│   ├── holdout_models/            # Checkpoints e scalers do treino final
│   ├── reports/                   # CSV/JSON de resultados
│   └── plots/                     # Figuras geradas
├── input/
│   └── nasa-battery-dataset/
│       └── cleaned_dataset/       # Dataset real esperado pelo pipeline
├── main.py                        # Pipeline principal
├── search.py                      # Busca de hiperparâmetros com Optuna
├── ood_val.py                     # Avaliação fora de domínio no frio
├── quantize.py                    # Quantização pós-treinamento (torchao)
├── check_splits.py                # Auditoria de particionamentos
├── run_pipeline.sh                # Orquestração shell do fluxo principal
├── pyproject.toml                 # Dependências declaradas
├── uv.lock                        # Lockfile do ambiente
└── .python-version                # Versão alvo do interpretador
```

---

## Tecnologias e dependências

### Ambiente base

- **Python 3.13** (`.python-version` e `requires-python = ">=3.13"`)
- Execução orientada a scripts, com saída persistida em `output/`

### Bibliotecas declaradas em `pyproject.toml`

- `altair`
- `ipykernel`
- `joblib`
- `matplotlib`
- `numpy`
- `optuna`
- `pandas`
- `polars`
- `pyarrow`
- `scikit-learn`
- `scipy`
- `seaborn`
- `snntorch`
- `spikingjelly`
- `tensorflow`
- `torch`
- `torchao`
- `torchvision`
- `tqdm`

### Stack técnica por componente

- **Processamento tabular / vetorial:** `polars`, `pandas`, `numpy`
- **Modelagem e treino:** `torch`, `torchvision`, `spikingjelly`, `snntorch`
- **Otimização de hiperparâmetros:** `optuna`
- **Métricas e splits:** `scikit-learn`
- **Quantização:** `torchao`
- **Visualização:** `matplotlib`, `seaborn`, `altair`
- **Leitura de dados/IO auxiliar:** `pyarrow`, `joblib`, `tqdm`

---

## Funcionalidades implementadas

### 1) Ingestão do dataset real da NASA

`data/ingestion.py` processa o dataset real a partir de:

- `input/nasa-battery-dataset/cleaned_dataset/metadata.csv`
- `input/nasa-battery-dataset/cleaned_dataset/data/*.csv`

O processamento atual faz o seguinte:

- filtra apenas ciclos do tipo `discharge`;
- exclui baterias listadas em `PipelineConfig.excluded_batteries`;
- aplica cutoff quando `Voltage_measured < 2.7V`;
- integra corrente para estimar capacidade e energia;
- calcula `SoH` com capacidade nominal fixa de 2.0 Ah;
- deriva `SoC`;
- extrai health indicators estáticos:
    - `HI_Time_of_Discharge`
    - `HI_Max_Temp`
    - `HI_Temp_Delta`
    - `HI_Mean_Temp`
    - `HI_Voltage_Integral`
    - `HI_Voltage_Drop`
- interpola cada ciclo para uma grade temporal uniforme de `time_steps`;
- salva o dataset consolidado em `output/processed/battery_health_dataset.csv`.

### 2) Construção de sequências temporais

`features/builder.py` transforma o dataframe processado em:

- `X`: tensor `[N, T, F]`
- `y`: alvo SoH
- `groups`: identificador da bateria para particionamento sem vazamento

Detalhes importantes:

- `Capacity_Ah`, `Energy_Wh` e `Delta_Q` são removidos das entradas do modelo para evitar leakage;
- apenas ciclos com exatamente `time_steps` amostras são mantidos;
- os grupos são definidos por `battery_id`.

### 3) Seleção automática de features

`features/selection.py` executa seleção por correlação de Spearman entre cada feature e o alvo `SoH`.

Regras atuais:

- a seleção é executada sobre o conjunto de desenvolvimento, e dentro de cada fold na cross-validation;
- features com correlação absoluta abaixo do limiar são descartadas;
- variáveis base físicas (`Voltage_measured`, `Current_measured`, `Temperature_measured`, `SoC`) são preservadas mesmo
  quando a correlação é baixa;
- a lista final é persistida em `output/reports/selected_features.json` e também por fold em
  `output/models/selected_features_foldN.json`.

### 4) Famílias de modelos implementadas

#### Modelos clássicos

Implementados em `models/classic_dl.py`:

- `CNN-1D`
- `LSTM`
- `CNN-LSTM`
- `CNN-DILATED`

#### Transformers e híbridos

Implementados em `models/transformers.py`:

- `iTransformer`
- `DynamicGraph-iTR`

#### Modelos fisicamente informados

Implementados em `models/physics.py`:

- `Phys-iTransformer`
- `Phys-iTR-Curriculum`

Esses modelos combinam a predição principal com um ramo físico derivado da corrente acumulada absoluta.

#### Modelos neuromórficos / SNN

Implementados em `models/spiking.py`:

- `SJ-Spiking-MultiStep-Decoupled`
- `SJ-Spiking-MultiStep`
- `SJ-Spiking-Attention`
- `SJ-Spiking-Hybrid`
- `SJ-Spiking-Simple`
- `SJ-LSM`
- `SJ-Spiking-Dilated`

Esses modelos utilizam componentes de `SpikingJelly`, neurônios LIF, atenção spiking e variações com codificação
temporal e reservatório esparso.

### 5) Fábrica central de modelos

`models/factory.py` instancia todas as arquiteturas acima com base em:

- `time_steps`
- `n_features`
- `d_model`
- `num_heads`
- `tau`
- `surrogate`
- `surrogate_alpha`

### 6) Loops de treino especializados

`training/loops.py` contém quatro rotinas principais:

- `train_standard`
- `train_physics`
- `train_curriculum`
- `train_spikingjelly`

Características do treino:

- `HuberLoss` como perda principal;
- `AdamW` como otimizador;
- `early stopping` baseado em validação;
- ruído gaussiano de regularização sobre os batchs de entrada;
- clipping de gradiente para modelos SNN;
- reset explícito de estados neuromórficos entre batches/avaliações.

### 7) Avaliação estruturada

#### Cross-validation

`evaluation/cross_val.py` executa:

- `StratifiedGroupKFold` com estratificação por domínio físico da bateria;
- split interno treino/validação via `GroupShuffleSplit`;
- escalonamento com `MinMaxScaler` em `X` e `y`;
- treinamento de todas as arquiteturas;
- salvamento de checkpoints, scalers e features selecionadas por fold.

#### Holdout

`evaluation/holdout.py` executa:

- divisão global de desenvolvimento vs. holdout;
- split interno treino/validação no desenvolvimento;
- treinamento final de todas as arquiteturas;
- avaliação no conjunto de teste inédito;
- persistência dos modelos finais e dos scalers globais.

#### OOD

`ood_val.py` define OOD como baterias classificadas como `Frio` em `BATTERY_DOMAINS` e testa o modelo em um cenário
zero-shot no frio (4°C), treinando somente em baterias não frias.

### 8) Quantização pós-treinamento

`quantize.py` usa `torchao` para quantização INT8 do readout dos modelos SNN, preservando o restante da arquitetura em
FP32.

O script:

- lê os checkpoints treinados;
- reconstrói o holdout ou os folds da cross-validation;
- carrega os scalers persistidos;
- quantiza apenas `fusion`, `regressor`, `fc_out` ou `readout`, quando disponível;
- mede tamanho do modelo, erro e tempo de inferência;
- salva `quantization_results.csv`.

### 9) Auditoria de splits

`check_splits.py` imprime o particionamento por baterias e domínios físicos para:

- holdout;
- treino final;
- validação final;
- cross-validation estratificada por fold.

---

## Pontos de entrada disponíveis

### Pipeline principal

Executa ingestão, seleção de features, cross-validation, holdout, plots e serialização da configuração final.

```bash
python main.py
```

### Busca de hiperparâmetros

Executa Optuna para a arquitetura `SJ-Spiking-MultiStep`.

```bash
python search.py
```

### Avaliação OOD

Executa o experimento fora de domínio no subconjunto frio.

```bash
python ood_val.py
```

### Quantização

Executa a comparação FP32 vs INT8 do readout dos modelos SNN.

```bash
python quantize.py
```

### Auditoria dos particionamentos

```bash
python check_splits.py
```

### Orquestração completa

`run_pipeline.sh` executa, nesta ordem:

1. `search.py`
2. `main.py`
3. `ood_val.py`

> Observação: o script **não** chama `quantize.py`. A quantização continua sendo uma etapa separada.

```bash
bash run_pipeline.sh
```

---

## Dados de entrada esperados

### Dataset real da NASA

O caminho padrão configurado em `PipelineConfig` é:

```text
./input/nasa-battery-dataset/cleaned_dataset
```

Esse diretório deve conter, no mínimo:

```text
metadata.csv
data/
  <arquivos de ciclos de descarga>
```

### Dataset sintético opcional

Se `input_dir` for apontado para `./input/synthetic_smartphones`, o projeto ativa o fluxo sintético e espera:

```text
smartphone_battery_dataset.csv
battery_cycle_summary.csv
```

---

## Artefatos gerados atualmente

### Processamento e relatórios

- `output/processed/battery_health_dataset.csv`
- `output/reports/selected_features.json`
- `output/reports/pipeline_config_final.json`
- `output/reports/best_hyperparameters.json`
- `output/reports/cv_summary_results.csv`
- `output/reports/cv_detailed_results.csv`
- `output/reports/holdout_results.csv`
- `output/reports/ood_results.csv`
- `output/reports/quantization_results.csv`
- `output/reports/optuna_full_history.csv`

### Checkpoints

- `output/models/`:
    - checkpoints por fold;
    - scalers por fold;
    - features selecionadas por fold;
    - modelos quantizados por fold.
- `output/holdout_models/`:
    - checkpoints finais;
    - scalers globais;
    - modelo quantizado final.

### Visualizações

- `output/plots/cross_val/`
- `output/plots/holdout/`

Os gráficos incluem, entre outros:

- curvas de loss por modelo e fold;
- comparações médias de MAE, RMSE e R²;
- radar chart;
- distribuição de erros;
- CED;
- heatmaps de erro;
- tendência temporal;
- scatter real vs. previsto.

---

## Configuração atual carregada pelo pipeline

Quando `output/reports/best_hyperparameters.json` existe, `PipelineConfig` sobrepõe automaticamente os defaults. O
arquivo atualmente armazenado contém:

```json
{
  "d_model": 4,
  "num_heads": 4,
  "tau": 5.0,
  "surrogate": "SoftSign",
  "surrogate_alpha": 6.0,
  "learning_rate": 0.0036533872806672426
}
```

Esses valores afetam principalmente as arquiteturas SNN e os Transformers que usam `d_model` e `num_heads`.

---

## Resultados registrados nos artefatos atuais

> Os números abaixo vêm dos CSVs já presentes em `output/reports/` e refletem o estado atual do projeto.

### Cross-validation

Fonte: `output/reports/cv_summary_results.csv`

| Model                | MAE_Mean           | MAE_Std             | RMSE_Mean          | RMSE_Std           | R2_Mean            | R2_Std               |
|----------------------|--------------------|---------------------|--------------------|--------------------|--------------------|----------------------|
| SJ-Spiking-MultiStep | 1.616983562707901  | 0.6924073076781201  | 2.54818205205042   | 1.2635219207908939 | 0.9521012008190155 | 0.04815623849452049  |
| SJ-Spiking-Attention | 6.101529002189636  | 1.3820106244700259  | 9.548528149454818  | 2.802931555279347  | 0.5207262933254242 | 0.2435336472512857   |
| SJ-Spiking-Hybrid    | 0.7433111220598221 | 0.19854897855615367 | 1.1942844481687547 | 0.3639163877536839 | 0.990999162197113  | 0.008023389098026865 |
| SJ-Spiking-Simple    | 4.350496292114258  | 0.6163987555480005  | 5.8319271786379066 | 1.1264020235349341 | 0.8292171061038971 | 0.054461174499731625 |
| SJ-LSM               | 3.852400481700897  | 1.3276120788361268  | 6.304439756613979  | 2.805934630167577  | 0.810741975903511  | 0.11561039562886874  |
| SJ-Spiking-Dilated   | 3.6168287992477417 | 2.7796288731930368  | 5.042851766799634  | 3.805007756284634  | 0.7166047692298889 | 0.41995954401689667  |
| CNN-1D               | 2.318231701850891  | 0.7309136138471009  | 3.127746254490984  | 0.9172056171236489 | 0.954618826508522  | 0.011897336680368025 |
| LSTM                 | 1.3550152778625488 | 0.5193874933179761  | 2.504996570075149  | 1.280427318966347  | 0.9678324311971664 | 0.023569233119200932 |
| CNN-LSTM             | 1.173190139234066  | 0.5135234497899555  | 1.9644172255244026 | 0.8990588940801469 | 0.9694717824459076 | 0.03127806497851961  |
| CNN-DILATED          | 1.8131912052631378 | 0.11146256632615656 | 2.46446946459438   | 0.5205475159085714 | 0.9661346524953842 | 0.019846473528059478 |
| iTransformer         | 2.804946333169937  | 1.364228368540341   | 4.3019079833305485 | 2.459408548597081  | 0.9059367477893829 | 0.08321902809530626  |
| DynamicGraph-iTR     | 3.175284743309021  | 1.0882855165187062  | 5.515647546385723  | 2.564763270468345  | 0.8537634611129761 | 0.08631273545543258  |
| Phys-iTR-Curriculum  | 2.520804703235626  | 1.511880567989537   | 3.831080740790794  | 2.3845552589762717 | 0.9135314524173737 | 0.07503094605034538  |
| Phys-iTransformer    | 3.1001071333885193 | 0.6741000507165344  | 4.920157197452793  | 1.7035308287971884 | 0.884620800614357  | 0.04786795753692273  |

Leitura prática: considerando não apenas o desempenho médio, mas também a estabilidade entre execuções (desvio padrão),
o modelo SJ-Spiking-Hybrid se destaca como o mais robusto e preciso, apresentando os menores erros e baixíssima
variância, seguido por CNN-LSTM e LSTM, que mantêm bom equilíbrio entre desempenho e estabilidade. O
SJ-Spiking-MultiStep, apesar de competitivo no cenário posterior, apresenta maior variabilidade e desempenho médio
inferior, enquanto modelos como CNN-1D e CNN-DILATED também demonstram boa consistência, porém com erros mais elevados.
Arquiteturas baseadas em Transformer, como iTransformer e suas variantes físicas, exibem maior variância e desempenho
inferior neste cenário.

### Holdout

Fonte: `output/reports/holdout_results.csv`

| Model                | MAE                | RMSE               | R2                 |
|----------------------|--------------------|--------------------|--------------------|
| SJ-Spiking-MultiStep | 0.506492555141449  | 0.7531574463878917 | 0.9983037710189819 |
| SJ-Spiking-Attention | 3.4989659786224365 | 4.535095267004933  | 0.9384979605674744 |
| SJ-Spiking-Hybrid    | 0.633913516998291  | 0.8339803369500273 | 0.9979201555252075 |
| SJ-Spiking-Simple    | 3.1526758670806885 | 4.195841210719988  | 0.947355329990387  |
| SJ-LSM               | 3.4857177734375    | 5.016578654956701  | 0.9247456192970276 |
| SJ-Spiking-Dilated   | 1.7527374029159546 | 2.3057371691012083 | 0.9841022491455078 |
| CNN-1D               | 3.8042259216308594 | 4.590738360288784  | 0.9369795322418213 |
| LSTM                 | 2.0170040130615234 | 2.4188018990402123 | 0.9825048446655273 |
| CNN-LSTM             | 0.904786229133606  | 1.0667289832409252 | 0.9965972900390625 |
| CNN-DILATED          | 1.4034277200698853 | 1.8185178234914496 | 0.9901109933853149 |
| iTransformer         | 1.0432432889938354 | 1.383678402241566  | 0.9942748546600342 |
| DynamicGraph-iTR     | 2.2503931522369385 | 2.6568156818093636 | 0.9788923859596252 |
| Phys-iTR-Curriculum  | 1.997101902961731  | 2.2903866632180514 | 0.9843131899833679 |
| Phys-iTransformer    | 1.172732949256897  | 1.5364774110116133 | 0.9929406046867371 |

Leitura prática: no holdout atual, os melhores resultados são dominados por arquiteturas spiking, com
SJ-Spiking-MultiStep como líder absoluto, seguido de SJ-Spiking-Hybrid. Entre os modelos não-spiking, destacam-se
CNN-LSTM, iTransformer e Phys-iTransformer, que apresentam desempenho consistente, porém inferior aos dois primeiros.

### OOD no frio

Fonte: `output/reports/ood_results.csv`

| Model                | MAE                | RMSE               | R2                  |
|----------------------|--------------------|--------------------|---------------------|
| SJ-Spiking-MultiStep | 7.190062046051025  | 8.419465861651071  | 0.5944579839706421  |
| SJ-Spiking-Attention | 14.886226654052734 | 18.098394400686832 | -0.8739020824432373 |
| SJ-Spiking-Hybrid    | 3.6540751457214355 | 4.380855456557543  | 0.8902044892311096  |
| SJ-Spiking-Simple    | 13.477045059204102 | 17.344153817089474 | -0.7209689617156982 |
| SJ-LSM               | 20.130861282348633 | 25.02577162857259  | -2.582958459854126  |
| SJ-Spiking-Dilated   | 3.7504353523254395 | 4.191732744084139  | 0.8994796276092529  |
| CNN-1D               | 8.940195083618164  | 11.102936543551742 | 0.2947508692741394  |
| LSTM                 | 3.210638999938965  | 4.094364498697369  | 0.9040952920913696  |
| CNN-LSTM             | 14.086286544799805 | 17.39502467790494  | -0.7310791015625    |
| CNN-DILATED          | 8.154223442077637  | 10.244809208699607 | 0.39955317974090576 |
| iTransformer         | 13.77912712097168  | 19.900066954533642 | -1.265561819076538  |
| DynamicGraph-iTR     | 21.79416847229004  | 25.496563848220926 | -2.719033718109131  |
| Phys-iTR-Curriculum  | 15.042778968811035 | 20.924119004238044 | -1.5047316551208496 |
| Phys-iTransformer    | 14.912175178527832 | 20.069401679897247 | -1.3042821884155273 |

Leitura prática: sob condições fora da distribuição (out-of-distribution), há uma mudança significativa no ranking, com
forte degradação de vários modelos — especialmente arquiteturas mais complexas e baseadas em Transformer. Os melhores
resultados passam a ser liderados por modelos com maior capacidade de generalização:

LSTM apresenta o melhor desempenho geral, com o menor MAE/RMSE e o maior R² (0.904), indicando maior robustez fora da
distribuição.
SJ-Spiking-Dilated e SJ-Spiking-Hybrid aparecem logo em seguida, com desempenho muito próximo e R² elevado (~0.89–0.90),
mostrando boa capacidade de generalização entre os modelos spiking.
SJ-Spiking-MultiStep, que era dominante in-distribution, sofre queda significativa (R² ≈ 0.59), evidenciando
sensibilidade a shift de distribuição.
Modelos como CNN-1D e CNN-DILATED mantêm desempenho intermediário, mas com perda relevante de precisão.
Arquiteturas como CNN-LSTM, iTransformer e variantes físicas apresentam colapso de generalização (R² negativo),
indicando falha em capturar padrões transferíveis para o domínio ODD.

Conclusão direta: em cenário ODD, modelos mais simples ou com vieses temporais mais diretos (como LSTM) e algumas
variantes spiking específicas (Hybrid/Dilated) são significativamente mais robustos do que arquiteturas mais
sofisticadas ou altamente especializadas.

### Quantização

Fonte: `output/reports/quantization_results.csv`

O relatório atual cobre apenas modelos SNN e compara FP32 vs INT8 no holdout e nos folds de cross-validation. O pipeline
de quantização quantiza somente o readout, o que explica a redução de tamanho modesta, mas com preservação parcial da
dinâmica interna.

---

## Limitações conhecidas e pontos de atenção

### 1) Assunções fortes de diretório

- `run_pipeline.sh` assume `./.venv/bin/python`.
- O dataset NASA precisa estar na estrutura esperada em `input/nasa-battery-dataset/cleaned_dataset/`.

### 2) Seleção de features e tamanho da janela

O pipeline filtra ciclos que não possuem exatamente `time_steps` amostras após a interpolação. Isso simplifica o formato
de entrada, mas pode descartar ciclos válidos com comprimentos diferentes.

### 3) OOD restrito a baterias frias

O experimento fora de domínio atual é específico para o domínio `Frio`. Ele é útil para stress test, mas não cobre todos
os cenários possíveis de mudança de distribuição.

### 4) Quantização parcial

A quantização não cobre as camadas recorrentes/atencionais internas; ela atua apenas no readout dos modelos SNN. Isso é
consistente com a intenção de preservar desempenho, mas limita a compressão total.

### 5) Hiperparâmetros e resultados podem mudar

Os relatórios em `output/reports/` são artefatos gerados. Se o pipeline for reexecutado com outra semente, outro dataset
ou novos hiperparâmetros, os números podem mudar.

---

## Como executar

### Pré-requisitos

- Python 3.13
- dataset NASA em `input/nasa-battery-dataset/cleaned_dataset/`
- ambiente com as dependências do projeto instaladas

### Execução dos principais fluxos

```bash
# Pipeline principal: ingestão, feature selection, CV, holdout e plots
python main.py

# Busca de hiperparâmetros do modelo SNN alvo
python search.py

# Experimento fora de domínio no frio
python ood_val.py

# Quantização pós-treinamento com torchao
python quantize.py

# Auditoria das divisões por bateria e domínio
python check_splits.py

# Orquestração completa disponível no repositório
bash run_pipeline.sh
```

### Fluxo recomendado de uso

1. Executar `search.py` para gerar `output/reports/best_hyperparameters.json`.
2. Executar `main.py` para processar dados, selecionar features, treinar modelos e gerar relatórios.
3. Executar `ood_val.py` para avaliar generalização fora de domínio.
4. Executar `quantize.py` quando os checkpoints e scalers já estiverem disponíveis.

---

## Reprodutibilidade

O projeto já contém mecanismos para tornar as execuções mais consistentes:

- `set_seed(42)` em `utils/common.py`;
- divisão por grupo baseada em `battery_id`;
- persistência de features selecionadas e scalers;
- serialização da configuração final em JSON.

Para reproduzir um experimento específico, mantenha:

- a mesma versão do dataset;
- o mesmo `best_hyperparameters.json`;
- os mesmos arquivos em `output/reports/` e `output/models/`;
- a mesma semente e o mesmo mapeamento de domínios físicos.

---

## Estado do projeto

O repositório está funcional como um pipeline experimental modular. Os componentes centrais de ingestão, treino,
avaliação, visualização e quantização existem e estão integrados, mas a documentação precisa acompanhar cuidadosamente
os artefatos gerados, pois alguns scripts operam sobre conjuntos específicos de modelos e algumas dependências ainda não
estão completamente formalizadas no `pyproject.toml`.

---

## Licença

Licença não especificada no repositório atual.