# Project Plan

## Contents

- [Etapas](#etapas)
    1. [Infraestrutura e Setup](#0-infraestrutura-e-setup)
    1. [Dados](#1-dados)
    2. [Modelagem](#2-modelagem)
    3. [Entrega Técnica](#3-entrega-técnica)
    4. [Apresentação](#4-apresentação)
    5. [Submissão](#5-submissão)

- Discussões

---

## Etapas

### 0. Infraestrutura e Setup

- [x] Fork do repositório + colaboradores no workspace #infra
- [x] Preencher README com nome da equipe e integrantes #infra (due 11/07)
- [x] Validar acesso ao Databricks Free Edition (notebook, catálogo, MLflow) #infra (due 11/07)
- [x] Definir convenção de branches (ex: feat/eda, feat/model, docs/slides) #infra (due 11/07)
- [x] Configurar .gitignore (ignorar *.csv grandes, mlruns/, .ipynb_checkpoints/) #infra (due 11/07)
- [x] Dividir as tarefas restantes entre os integrantes (definir responsável por item) #infra (due 11/08)
- [x] Escrever documento de propósito: objetivo, pergunta principal e escopo (priorizar o que foi pedido, curiosidade fica para depois) #infra (due 11/09)
- [x] Subir a ata/resumo da reunião de 11/07 no GitHub (.md) #infra (due 11/08)


### 1. Dados

- [x] Carregar `history.csv` e `inference.csv `no workspace #data (due 11/07)
- [x] EDA inicial: shape, tipos, missing, distribuição do target Churn #data (due 11/07)
    - **Achados preliminares (Bruno)**: contrato mensal ~50% de churn vs. 2 anos quase nenhum; clientes novos churnam mais; uma forma de pagamento com churn maior; gênero, idoso e parceiro sem diferença relevante
- [x] Tratar TotalCharges (valores em branco/string) #data (due 11/07)
    - **Decisão adotada**: Modelo de regressão Ridge treinado e registrado como champion imputer no MLflow; Bruno validou fórmula empírica `tenure * MonthlyCharges`.
- [x] Investigar divergência de linhas vs. slide introdutório (history 5634/5912, inference 1409/1461) e confirmar com os organizadores #data (due 11/08)
- [x] Verificar sobreposição de clientes entre `history.csv` e `inference.csv` #data (due 11/08)
    - Verificado no `03.0_ETL_sql.ipynb`: sobreposição nula de customerID.
- [x] Inspecionar `inference.csv` atrás de "pegadinhas" (grafias, categorias novas, nulos que não existem no treino) #data (due 11/08)
    - Existe a variável `CustomerFeedback`, que contém explicações como "decided to cancel"; isso indica que a **variável está vazada** e não deve ser usada para prever o modelo.
    - Todas as demais colunas estão presentes e sem novas categorias.
- [x] Corrigir grafia de PhoneService (Yes/No/yes/no) e converter para binário #data (due 11/08)
- [x] Tratar os 19 "unknown" de `tenure` (checar se coincidem com os nulos de TotalCharges antes de usar regressão: risco de imputar uma pela outra) #data (due 11/08)
    - Tratado via `TRY_CAST(tenure AS BIGINT)` no SQL e imputação por mediana no pipeline.
- [x] Checar multicolinearidade TotalCharges × tenure (correlação ~0.82-0.84) e decidir manter uma ou as duas (comparar no modelo) #data (due 11/08)
    - Avaliado na ABT1 / ABT2 e no modelo Ridge; correlações mapeadas.
- [x] Tratar categóricas (One-Hot ou Target Encoding) #data (due 11/08)
    - Criar flags binárias has_internet e has_phone; tratar "No internet service" / "No phone service" como "No" nas colunas de serviços; fazer drop de uma dummy (checar se "No phone service" existe em MultipleLines)
- [x] Processar CustomerFeedback (TF-IDF + SVD ou features simples) #data (due 11/08)
    - [x] Teste de vazamento do `CustomerFeedback`: taxa de frases de saída/permanência em history (por `TargetChurn`) vs. inference (SQL) + AUC de modelo só-texto #data (due 11/08)
    - [x] Documentar no README e em 1 slide: `CustomerFeedback` fora do modelo de churn (vazamento: o comentário descreve o desfecho e não existiria para um cliente real no momento da ligação) #data (due 11/08)
    - [x] Extrair temas dos comentários / Análise de sentimento (Thiago) (preço/valor, concorrente, forma de pagamento, mudança de endereço, qualidade) e comparar churners vs. stayers; separar motivos acionáveis de não acionáveis #data (due 11/09)
    - [x] Avaliar usar o texto para preencher os "unknown" de `tenure`: comparar tenure/`MonthlyCharges` citados no texto com as colunas onde ambos existem; só usar se concordarem #data (due 11/08)
- [x] Split treino/validação estratificado + pipeline reprodutível (sklearn Pipeline) #data (due 11/08)
    - Imputação calculada só com dados de treino; regra padrão para nulos na inferência (numéricas: média/zero; categóricas: moda); mesmo código para treino e inferência
- [x] Checar desbalanceamento do target e decidir se/como balancear #data (due 11/08)
- [x] Verificar se existe dicionário de dados; se não, montar um (ex: significado de DSL/fibra, "unknown") #data (due 11/08)
    - Documentado no `06_data/README.md` e em `03.0_ETL_sql.ipynb`.
- [x] Consolidar perguntas e enviar ao Prof (nulos de TotalCharges, "unknown" em tenure, por que clientes ficam no plano mensal, dicionário de dados, divergência de linhas, se o `CustomerFeedback` da inferência deveria declarar o desfecho) #data (due 11/09)

### 2. Modelagem

- [x] Baseline: Logistic Regression + métricas (AUC, Recall, F1) #model (due 11/07)
    - Concluído no baseline de Bruno e no benchmark MLflow de Raquel.
- [x] Modelo principal: LightGBM (ou XGBoost CPU) com CV estratificada #model (due 11/08)
    - LightGBM treinado e selecionado como champion com CV estratificada em ambos os trabalhos.
- [x] Decidir tenure contínuo vs. grupos (novo/médio/antigo) e se haverá modelos separados por tempo de casa #model (due 11/08)
    - Decidido manter contínuo + faixas/bandas de tempo de casa (`isTenureLow`, `ChargeBand`) para explicabilidade.
- [x] Confirmar métricas de seleção (AUC, Recall, F1) e política de corte (threshold, top-N ou prob. > 90%) com o time #model (due 11/08)
- [x] Ajuste de threshold focado em Recall (justificar custo de falso negativo) #model (due 11/08)
    - Avaliado corte ótimo e gerados arquivos de probabilidade e corte (`predictionPCT` e `prediction_Gt`).
- [x] Registrar experimentos no MLflow (parâmetros, métricas, artefato do modelo) #model (due 11/08)
    - Registrados os experimentos de TotalCharges imputer e Churn classifier.
- [x] Salvar o pipeline completo (pré-processamento + modelo) como um único artefato no MLflow, reutilizado pelo `prediction.csv` e pelo App #model (due 11/09)
- [x] Rodar primeiro modelo com todas as variáveis e podar depois (coeficientes/significância/importância) #model (due 11/09)
- [x] Análise de feature importance + interpretação de negócio #model (due 11/09)
    - Salvo `df_importances.csv` e analisados principais drivers (tenure, contract, payment method).
- [x] Explicação por cliente (principais drivers do score: coeficientes/SHAP) para indicar "sobre o que" vem o risco, sem usar o texto #model (due 11/10)
- [x] Análises estatísticas de apoio (associação churn × contrato, tempo de casa, forma de pagamento; qui-quadrado/correlações) #model (due 11/09)
    - Análise detalhada no notebook de Bruno e no `03.0_ETL_sql.ipynb`.
- [x] Redigir recomendações de negócio de retenção: onde atuar e em quais perfis (ex: incentivar planos longos; não oferecer grandes descontos a quem já tem prob. muito alta) #model (due 11/10)
- [x] (nice-to-have) Comparar com modelo alternativo (ex: Random Forest) #model nice-to-have (due 11/10)
    - 7 modelos comparados (Logistic Regression, Random Forest, LightGBM, XGBoost, MLP, Decision Tree, KNN).
- [x] Documentar justificativa da escolha do modelo final (comparar métricas, não só acurácia; mostrar que os demais foram testados) #model (due 11/10)

### 3. Entrega Técnica

- [x] Gerar `prediction.csv` a partir de `inference.csv`: probabilidade de churn por cliente (permite testar cortes diferentes depois) #delivery (due 11/09)
    - Gerados `predictionPCT_Group3_20261112_052957.csv` e `prediction_Gt_20261112_052538.csv`.
- [x] Validar formato do `prediction.csv` (colunas esperadas, mesmo número de linhas e IDs da `inference.csv`, sem NaN, probabilidades em [0,1]) #delivery (due 11/09)
- [x] Subir `prediction.csv` no repo (ou link no README) #delivery (due 11/09)
- [x] Definir escopo mínimo do App: selecionar cliente existente (prob. + recomendação) vs. input manual de features #delivery (due 11/08)
    - Definido: Dashboard no Databricks.
- [x] Desenhar regras de recomendação por faixa de probabilidade e tempo de casa (o que o atendente deve oferecer) #delivery (due 11/09)
- [x] Salvar probabilidades da base de inferência em tabela para alimentar o App / Dashboard #delivery (due 11/09)
- [ ] Construir Databricks Dashboard (Madu e Carol): input/seleção de cliente → visualização de probabilidade, drivers de risco e sugestão de retenção #delivery (due 11/10)
- [ ] Testar Dashboard no Free Edition (confirmar atualização e filtros) #delivery (due 11/10)
- [x] (fallback) Notebook com widgets interativos caso o App falhe #delivery (due 11/10)
- [x] README final: arquitetura, como rodar, decisões técnicas #delivery (due 11/12)
    - Consolidado em `ORGANIZED/README.md`.
- [x] (nice-to-have) API com Model Serving #delivery nice-to-have (só se sobrar tempo)
- [ ] (nice-to-have) Chatbot com Genie/Custom Agents #delivery nice-to-have (só se sobrar tempo)

### 4. Apresentação

- [x] Definir narrativa: problema → dados → modelo → produto → impacto de negócio #presentation (due 11/11)
    - Estrutura acordada (DEC-014, Reunião 09/11): 10 slides (10 min), 3 min demo ao vivo no Databricks, 2 min de perguntas.
- [x] Montar slides (15 min, foco em negócio para banca mista) #presentation (due 11/11)
- [ ] Preparar demo do App / Dashboard (roteiro de 2-3 min) #presentation (due 11/11)
- [x] Preparar respostas para perguntas prováveis (por que não Model Serving? por que Recall? como o time de vendas usa?) #presentation (due 11/11)
- [ ] Ensaio cronometrado (15 min) #presentation (due 11/12)
- [ ] Gravar vídeo backup da demo (caso o App caia na hora) #presentation (due 11/12)

### 5. Submissão

- [x] Revisar estrutura do repo (pastas, README, notebooks limpos) #infra (due 11/12)
    - Concluído com a estrutura padronizada em `ORGANIZED/`.
- [ ] Garantir todos os commits antes de ==13/11 10:00== #infra (due 11/12)
- [x] Preencher Google Forms com link do repo + `prediction.csv` #infra (due 11/12)
- [ ] Confirmar submissão (print/screenshot do Forms enviado) #infra (due 11/12)

---

## Discussões

> Q = Question |  A = Answer | D = Decision

### Reunião do time: 2026-11-09
> Duração: ~2h \
> Presentes: Bruno, Caroline, Maria Eduarda, Raquel, Thiago

- **Modelagem de TotalCharges (Raquel & Bruno):**
  - D: Modelo Ridge (`alpha=100`, GridSearchCV) treinado e registrado como champion no MLflow ($R^2 \approx 0.999$, RMSE $\approx \$65$).
  - D: Remoção de variáveis collineares mantendo correlação $\ge 0.15$; dominância de `TotalSpend` (99.9%) validada como física da cobrança (DEC-008).
  - D: Nenhuma linha pode ser descartada na inferência (todos os 1.409 clientes precisam de score).
- **Tratamento de Missing e Tenure:**
  - D: Bruno validou `tenure = ROUND(TotalCharges / MonthlyCharges)`, resolvendo 18 dos 19 casos sem tenure.
  - D: Para o caso de ausência dupla (linha 1097), usar mediana do contrato como placeholder temporário com flag, rodar o imputer Ridge e recalcular o tenure ao final (cascata sanduíche DEC-009).
  - D: Prevenção de vazamento de dados (preocupação do Bruno resolvida em DEC-011): toda imputação encapsulada em `Pipeline` do Scikit-Learn com `FunctionTransformer`.
- **Análise de Sentimento e Motivos de Churn (Thiago):**
  - D: Classificador baseado em regras contrastivas (ponderação pós-"mas") extrai motivos: Rede (instabilidade + velocidade), Preço, Concorrente e Forma de Pagamento.
  - D: `CustomerFeedback` e os motivos extraídos **NÃO** entram como features no modelo de churn (evita 99% de acurácia artificial por vazamento, DEC-001 / DEC-012).
  - D: Ideia aceita de modelos especializados por motivo acionável (Preço, Concorrente, Pagamento) para guiar as ações do operador de retenção (DEC-013).
  - Pendência: Raquel validar manualmente uma amostra de 30 feedbacks.
- **Dashboard e Arquitetura no Databricks (Carol, Madu, Thiago):**
  - D: Dashboard construído com Databricks SQL via tabelas/views no catálogo (sem campos calculados na ferramenta de BI).
  - D: Visão por cliente (jornada/ID lookup): exibir forma de pagamento, tenure, monthly charges, score de churn e recomendação de ação.
- **Estratégia da Apresentação (15 min):**
  - D: 10 slides focados em decisões críticas (TotalCharges, cascata de tenure, governança de vazamento), 3 min de demo ao vivo do app/dashboard e 2 min de Q&A (DEC-014).
  - D: Slide dedicado à estrutura da ABT (requisito dos avaliadores) e gráfico de feature importance focado no top 10.
- **Organização do Repositório:**
  - D: Entrega em repositório único centralizado na `main`, pastas organizadas por frente (modelos, sentimento, dashboard, documentação).

---

### Reunião do time: 2026-11-08
> Duração: ~2h \
> Presentes: Bruno, Caroline, Raquel, Thiago

- D: `CustomerFeedback_adj` tratado como **vazamento**: nos exemplos, o comentário declara o desfecho ("decided to churn", "recently canceled" vs. "no plans to switch") e repete dados tabulares (tenure, `MonthlyCharges`, pagamento, internet, contrato). Um comentário (provavelmente da inferência) também declara o cancelamento. Fica fora do modelo de churn e do App; uso apenas descritivo (temas/motivos).
    - A confirmar com SQL: taxa de frases de saída/permanência por dataset e por classe.
- Obs: nem todo churn é acionável (ex: mudança para área sem cobertura); o App não deve sugerir desconto nesse caso.
- Obs: sentimento separa pouco churners de stayers (churners dizem "satisfied overall" e saem); temas são mais úteis.
- D: `prediction.csv` = probabilidade de churn por cliente (assumido), o que permite definir cortes/thresholds diversos depois.
- D: Riscos de prazo aceitos pelo time (deploy tarde, empacotamento do modelo, corte de commits, data da apresentação): o cronograma é apertado e será cumprido correndo.

---

### Reunião do time: 2026-11-07
> Duração: ~2h \
> Presentes: Bruno, Caroline, Maria Eduarda, Raquel, Thiago

- **Objetivo**: modelo de probabilidade de churn (prioridade) + recomendações de negócio + solução ponta a ponta (dashboard onde se seleciona o cliente e vê probabilidade e recomendação). Não sugerir grandes descontos para quem já tem prob. muito alta.
- D: GitHub como repositório central, documentação em `.md`, cada pessoa na sua branch/workspace.
- D: Próxima reunião domingo 11/08 às 17h; meta: ao menos um modelo pronto.
- D: Base de treino dividida em treino/teste com validação cruzada; base de inferência com a mesma estrutura, reaproveitando as transformações (ETL reutilizável).
- D: Primeiro modelo com todas as variáveis (correlação baixa na maior parte); podar depois se necessário.
- D: Codificação: gênero 0/1, pagamento em one-hot, flags `has_internet`/`has_phone`, "sem serviço" tratado como "não" nas colunas de produtos.
- D: Comparar regressão logística com modelos de árvore/ensemble; justificar a escolha por métricas.
- D: Levar dúvidas de negócio ao Carlos após a EDA.
- Verificado: todo cliente sem internet aparece como "sem serviço de internet" em todas as colunas de produtos (sem inconsistência).
- Q: Imputação em produção: calcular com dados de treino; regra padrão para nulos na inferência (numéricas média/zero, categóricas moda). Descartar cliente não é opção.
- Q: Tenure em grupos (novos/médios/antigos) com modelos separados, ou contínuo? Em aberto.
- Q: Balanceamento de classes? Depende da EDA.
- Q: Corte do modelo (top 20, > 90%?) e métricas de seleção. Em aberto.
- Q: Por que clientes ficam no plano mensal? (falha em oferecer plano longo, preferência por não ficar preso, multa?) Perguntar ao Carlos.
- Q: Quem faz o deploy do App e em qual conta (limite de 1 app por conta)? Em aberto.



- Quantidade de linhas dos arquivos vs slide introdutório:
    - `history.csv` 5634 / 5912 *(-278)*
    - `inference.csv` 1409 / 1461 *(-52)*

<br>

- Variáveis:
    - `TotalCharges` com 207 missings
        - Q: Cálculo reverso usando `tenure` e `MonthlyCharges`?
            - A: Alguns testes feitos e vemos que os valores não batem na comparação `tenure * MonthlyCharges = TotalCharges`, não dá para simplesmente "substituir".
            - A: `TotalCharges` e `tenure` tem alta correlação de Peason, talvés só uma seja mais importante ter no modelo.
        - Q: Qual valor imputar? Média? Regressão Linear para estimar baseado em outras variáveis?
        - Q: Remoção pode ser uma opção?
            - A: Se quisermos escorar todos os clientes e manter essa base o ideal não seria remover, pois se esse cliente ligar não teremos a probabilidade de churn já calulada e não teriamos a melhor abordagem para manter o cliente.
    - `tenure` variável numérica com uma categoria "unknown".
        - Q: O que fazer quando não temos quanto tempo o cliente está na base?
            - A: Como também é numerica, e ela é altamente correlacionada com `TotalCharges`, podemos tentar Regressão Linear.
        - Q: Podemos usar como categoria para ter modelos diferentes para clientes Novos/Antigos?
        - Q: Pode ser útil após uam analise de sentimento sobre porque pessoas em diferentes períodos decidem cancelar.
    - `PhoneService` variável categorica com 4 categorias: Yes/No/yes/no
        - D: Correct spelling to transform into binary.
        - Q: Note that we don't have "No phone servive" as a category.
    - `CustomerFeedback` variável de texto podendo ser útil para definir o porquê o consumidor decidiu sair.
        - Q: Análise de sentimento para criar possíveis categorias do porquê do churn.

---

<br>

- Conceitos
    - A companhia oferece pelo menos 2 serviços: **internet** ou **phone**, e podemos ter clientes com somente 1 ou os dois.
        - Q: Definir quais variáveis são de qual serviço para tentar resolver as categorias "No phone service" / "No internet service".  Assim podemos criar uma variável para para clientes com telefone (binária), outra para clientes com intenet (binária) e verificar se "No phone service" é consistente com clientes que não tem linha de telefone.
