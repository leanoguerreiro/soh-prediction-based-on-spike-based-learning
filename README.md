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
    - `HI_Thermal_Integral`
    - `HI_Temp_Rate`
    - `HI_Voltage_Efficiency`
    - `HI_Time_to_Max_Temp`
    - `HI_Final_Voltage`
    - `HI_Internal_Resistance`
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

| Model                | MAE_Mean           | MAE_Std             | RMSE_Mean          | RMSE_Std           | R2_Mean            | R2_Std                |
|----------------------|--------------------|---------------------|--------------------|--------------------|--------------------|-----------------------|
| SJ-Spiking-MultiStep | 1.465942144393921  | 0.5369203550381392  | 2.163338935490973  | 0.7093067139927979 | 0.9706299304962158 | 0.02443097050354622   |
| SJ-Spiking-Attention | 4.07724928855896   | 0.4537730827921739  | 6.458553823747127  | 1.1933924696054048 | 0.7621074765920639 | 0.15691747275857745   |
| SJ-Spiking-Hybrid    | 1.4525528997182846 | 0.6618262364492865  | 2.236924739261464  | 1.1282805284789112 | 0.9564872980117798 | 0.05406528554920556   |
| SJ-Spiking-Simple    | 3.6086349487304688 | 0.8407053453822877  | 4.729248593051048  | 1.0358850851026529 | 0.868026003241539  | 0.0965238470887006    |
| SJ-LSM               | 4.221303939819336  | 1.5993104539236225  | 6.709421728024378  | 3.387508397580054  | 0.77812559902668   | 0.16628883049899612   |
| SJ-Spiking-Dilated   | 1.4388587176799774 | 0.2888846304400355  | 1.6885702011452628 | 0.3171314522400927 | 0.9851776361465454 | 0.007003845539043493  |
| CNN-1D               | 1.9004437625408173 | 0.5203177559742033  | 2.5943318851863193 | 0.7490693835308017 | 0.9686937481164932 | 0.00833698352583511   |
| LSTM                 | 1.235514760017395  | 0.33478181161473497 | 1.8553799252486276 | 0.4820963591830097 | 0.9839676916599274 | 0.0030571456343277373 |
| CNN-LSTM             | 1.3935880959033966 | 0.3162505355635853  | 2.5831274732093283 | 0.4907081778509408 | 0.9659289419651031 | 0.012973002952087838  |
| CNN-DILATED          | 1.7806618809700012 | 0.6188159018049607  | 3.0959197428156635 | 1.440490393079668  | 0.9302978366613388 | 0.06416996509241359   |
| iTransformer         | 3.23615962266922   | 0.6598160728411796  | 5.559500549349493  | 1.2808455342908713 | 0.832585945725441  | 0.0911671923695662    |
| DynamicGraph-iTR     | 3.0377995669841766 | 1.029995819024492   | 4.9917527292636255 | 2.303942534505184  | 0.8576802760362625 | 0.1022477234454821    |
| Phys-iTR-Curriculum  | 2.692037045955658  | 0.9523926064620195  | 3.3899660599832067 | 1.023273502862967  | 0.9253071695566177 | 0.05371692492142196   |
| Phys-iTransformer    | 3.4666662514209747 | 1.0651397001507503  | 5.059782740170305  | 1.643890821262946  | 0.863190159201622  | 0.061473529145225324  |

Leitura prática: ao considerar simultaneamente desempenho médio e estabilidade (desvio padrão), observa-se uma mudança
relevante no ranking. O modelo SJ-Spiking-Dilated passa a apresentar o melhor equilíbrio entre precisão e consistência,
com baixa variabilidade entre execuções e alto poder explicativo. Em seguida, o LSTM se destaca como o modelo mais
estável estatisticamente, mantendo erros baixos e variância mínima.

Modelos como SJ-Spiking-MultiStep e SJ-Spiking-Hybrid permanecem competitivos em termos de desempenho médio, porém
apresentam maior sensibilidade às partições dos dados, refletida em desvios padrão mais elevados. Arquiteturas como
CNN-1D e CNN-LSTM continuam sendo alternativas sólidas, com desempenho consistente, embora inferiores aos melhores
casos.

Por outro lado, modelos baseados em Transformer (como iTransformer e suas variações) mantêm desempenho inferior e maior
variabilidade, indicando menor robustez no cenário de validação cruzada.

### Holdout

Fonte: `output/reports/holdout_results.csv`

| Model                | MAE                | RMSE               | R2                 |
|----------------------|--------------------|--------------------|--------------------|
| SJ-Spiking-MultiStep | 0.9997504353523254 | 1.4047568553398153 | 0.9940990805625916 |
| SJ-Spiking-Attention | 1.8903720378875732 | 2.435030737962915  | 0.9822693467140198 |
| SJ-Spiking-Hybrid    | 1.069430947303772  | 1.3643888330042313 | 0.9944333434104919 |
| SJ-Spiking-Simple    | 2.113936424255371  | 2.696830079023922  | 0.9782517552375793 |
| SJ-LSM               | 3.004192590713501  | 3.9537661790109713 | 0.9532546401023865 |
| SJ-Spiking-Dilated   | 1.5758343935012817 | 1.8753516821064762 | 0.9894832372665405 |
| CNN-1D               | 1.6323901414871216 | 2.0121755729424895 | 0.9878926873207092 |
| LSTM                 | 1.635323405265808  | 1.952881271413766  | 0.988595724105835  |
| CNN-LSTM             | 0.9930282831192017 | 1.2396463276646956 | 0.9954047203063965 |
| CNN-DILATED          | 2.0072572231292725 | 2.5607513415932597 | 0.9803912043571472 |
| iTransformer         | 2.4356911182403564 | 3.0239607483874664 | 0.9726555943489075 |
| DynamicGraph-iTR     | 4.453719615936279  | 5.576516141134971  | 0.9070086479187012 |
| Phys-iTR-Curriculum  | 2.8703033924102783 | 3.7585344473633073 | 0.9577571153640747 |
| Phys-iTransformer    | 4.227468013763428  | 5.442152115966652  | 0.9114358425140381 |

Leitura prática: no cenário holdout (in-distribution), o desempenho geral é elevado para a maioria dos modelos, com
diferenças mais sutis entre os melhores. O modelo CNN-LSTM apresenta o melhor resultado global, alcançando o menor erro
e maior capacidade explicativa.

As arquiteturas spiking SJ-Spiking-MultiStep e SJ-Spiking-Hybrid aparecem logo em seguida, com desempenho muito próximo
e altamente competitivo. Modelos como LSTM e CNN-1D mantêm desempenho sólido e consistente, configurando bons baselines.

Já SJ-Spiking-Dilated, embora robusto em validação cruzada, apresenta desempenho ligeiramente inferior neste cenário
específico. Modelos baseados em Transformer continuam apresentando resultados inferiores em comparação às demais
abordagens.

### OOD no frio

Fonte: `output/reports/ood_results.csv`

| Model                | MAE                | RMSE               | R2                  |
|----------------------|--------------------|--------------------|---------------------|
| SJ-Spiking-MultiStep | 5.155243396759033  | 7.299657254468516  | 0.6951601505279541  |
| SJ-Spiking-Attention | 8.167725563049316  | 9.540021418581416  | 0.47932642698287964 |
| SJ-Spiking-Hybrid    | 4.8495049476623535 | 6.605055820364936  | 0.750414252281189   |
| SJ-Spiking-Simple    | 29.50758934020996  | 39.06168280395204  | -7.7290849685668945 |
| SJ-LSM               | 23.515832901000977 | 26.384731431916737 | -2.9826505184173584 |
| SJ-Spiking-Dilated   | 10.465200424194336 | 12.387000844312242 | 0.12219280004501343 |
| CNN-1D               | 14.409918785095215 | 16.405563601712796 | -0.5397460460662842 |
| LSTM                 | 19.047870635986328 | 24.731806069216063 | -2.4992783069610596 |
| CNN-LSTM             | 9.19282054901123   | 12.695579925067978 | 0.07791298627853394 |
| CNN-DILATED          | 14.41483211517334  | 16.573523196429242 | -0.5714352130889893 |
| iTransformer         | 15.971542358398438 | 21.741920110420114 | -1.7043483257293701 |
| DynamicGraph-iTR     | 17.27630043029785  | 19.246987639709364 | -1.1192996501922607 |
| Phys-iTR-Curriculum  | 13.271665573120117 | 18.188251702678848 | -0.8925559520721436 |
| Phys-iTransformer    | 19.626596450805664 | 23.051413747994935 | -2.0399186611175537 |

Leitura prática: sob condições fora da distribuição (out-of-distribution), ocorre uma degradação significativa e
generalizada no desempenho dos modelos, com mudanças expressivas no ranking. Nesse cenário, as arquiteturas spiking mais
estruturadas passam a dominar.

O modelo SJ-Spiking-Hybrid apresenta o melhor desempenho geral, seguido por SJ-Spiking-MultiStep, ambos demonstrando
maior capacidade de generalização sob distribuição deslocada. O modelo SJ-Spiking-Attention ainda mantém desempenho
utilizável, embora com erro elevado.

A partir desses, observa-se uma queda acentuada: modelos como SJ-Spiking-Dilated e CNN-LSTM apresentam baixo poder
explicativo, enquanto LSTM, CNN-1D, CNN-DILATED e arquiteturas baseadas em Transformer exibem colapso de generalização,
frequentemente com R² negativo. Casos mais simples, como SJ-Spiking-Simple e SJ-LSM, apresentam degradação severa,
indicando incapacidade de adaptação ao novo domínio.

Conclusão direta: no cenário OOD, modelos com viés estrutural mais forte e regularização implícita (como as arquiteturas
spiking Hybrid e MultiStep) demonstram maior robustez, enquanto modelos mais complexos ou altamente ajustados ao domínio
de treino tendem a falhar na generalização.

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