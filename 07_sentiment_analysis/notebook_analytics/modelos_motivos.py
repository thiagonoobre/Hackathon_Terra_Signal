# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Probabilidade de saída POR MOTIVO (apoio ao operador)
# MAGIC
# MAGIC **Ideia:** o texto ensina, as características do cliente preveem.
# MAGIC - O **rótulo** de cada motivo vem do `CustomerFeedback` do HISTORY (regras do notebook `motivos_cancelamento`).
# MAGIC - As **features** são só as colunas tabulares. Nada de texto, então não há vazamento.
# MAGIC - Assim o modelo funciona para QUALQUER cliente do inference, inclusive quem não escreveu que vai sair.
# MAGIC
# MAGIC **Conta final, para cada motivo m:**
# MAGIC `P(sair por m) = P(churn) × P(m | churn)`
# MAGIC - `P(churn)`: modelo treinado em toda a base (alvo = Churn)
# MAGIC - `P(m | churn)`: modelo treinado SÓ em quem saiu (alvo = o motivo m apareceu no texto?)
# MAGIC
# MAGIC Pré-requisito: rodar antes `%run ./motivos_cancelamento` (função `processar`).

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import re
import pandas as pd

# COMMAND ----------


df_hist = (spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiLine", "true")   # permite quebras de linha dentro de aspas
      .option("escape", '"')         # trata "" dentro do texto como aspas literais
      .load("/Volumes/workspace/asn/metrics/history.csv"))


df_infer = (spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiLine", "true")   # permite quebras de linha dentro de aspas
      .option("escape", '"')         # trata "" dentro do texto como aspas literais
      .load("/Volumes/workspace/asn/metrics/inference.csv"))

df_history = df_hist.toPandas()
df_inference = df_infer.toPandas() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Motivos: o assunto aparece EM TOM DE RECLAMAÇÃO?
# MAGIC Cada texto é quebrado em **trechos**: primeiro em frases, depois em cada "but", "while",
# MAGIC "however", "although", ";". Um motivo só conta se o MESMO trecho:
# MAGIC 1. fala do assunto (palavras do tema), **e**
# MAGIC 2. tem palavra de queixa ("too high", "slow", "outages", "inconvenient"...).
# MAGIC
# MAGIC Exemplo: "While the fiber was fast and reliable, | I found the charges too high"
# MAGIC vira 2 trechos. Só o segundo tem queixa, então o motivo é **preço**, e não instabilidade.

# COMMAND ----------

# Frases em que o cliente afirma a decisão de sair
DECLAROU = re.compile(
    r"decided to (?:churn|cancel|switch|leave|discontinue|end|terminate)"
    r"|(?:have|had|ultimately|finally) decided"
    r"|deciding to (?:churn|cancel|switch|leave)"
    r"|ended up (?:churn|cancel|switch|leav)"
    r"|had to (?:churn|cancel|switch|leave|discontinue)"
    r"|chose to (?:churn|cancel|leave|switch)"
    r"|(?:made the )?decision to (?:churn|cancel|leave|discontinue)"
    r"|i (?:have )?(?:churned|cancel+ed|switched|left)"
    r"|recently (?:cancel+ed|switched|decided)"
    r"|i am (?:churning|leaving|switching|cancel+ing)"
    r"|plan(?:ning)? to (?:churn|cancel|switch|leave)",
    re.I)

# Frases que NEGAM a saída ou falam de ter vindo PARA esta empresa
NEGA = re.compile(
    r"no (?:plans|intention) to (?:churn|switch|leave|cancel)"
    r"|not (?:planning|plan) to|do not plan|don.t plan|never considered"
    r"|would not (?:switch|leave)|no reason to (?:switch|leave|churn)"
    r"|switch(?:ed)? to this provider",
    re.I)

# Palavras de insatisfação forte (para quem reclama sem dizer que vai sair)
INSATISFEITO = re.compile(
    r"disappointed|dissatisfied|frustrat|unacceptable|terrible|poor service"
    r"|not (?:happy|satisfied)|considering (?:switching|other|leaving|cancel)",
    re.I)

def status_texto(t):
    if DECLAROU.search(t) and not NEGA.search(t):
        return "1_declarou_saida"
    if INSATISFEITO.search(t):
        return "2_insatisfeito"
    return "3_satisfeito"

# COMMAND ----------

MOTIVOS = {
    "preco":         r"charge|price|cost|expensive|afford|fee|pricing|\$\d",
    "aumento_preco": r"increase|hike|raised|went up",
    "instabilidade": r"outage|disconnect|dropp|unreliab|reliab|connection|connectivity|interruption|disruption|downtime",
    "velocidade":    r"slow|speed",
    "pagamento":     r"check|payment method|mail(?:ed|ing)|payment option",
    "atendimento":   r"customer service|support|representative|staff|agent",
    "cobranca":      r"billing|overcharg|invoice",
    "sem_internet":  r"lack of internet|no internet|without internet|not received any connection|did not (?:have|include) internet",
    # só conta quando há uma OFERTA melhor; "switch to a different provider" sozinho é o destino, não o motivo
    "concorrente":   r"better (?:deal|offer|value|price|pricing|plan|rate)|cheaper|lower price|more affordable|competitor",
    "contrato":      r"flexible contract|locked into|contract (?:terms|length)",
    "pessoal":       r"personal reasons|financial|moving|relocat|no longer (?:need|require)",
}
QUEIXA = re.compile(
    r"too |steep|\bhigh(?:er)?\b|expensive|inconvenien|outdated|hassle|issue|problem|unreliab|not reliable"
    r"|slow|frequent|lack|poor|not (?:worth|meet|up to|always)|struggl|disappoint|frustrat"
    r"|difficult|old-fashioned|increase|hike|better|cheaper|cumbersome|annoy|inconsisten"
    r"|outage|interruption|disruption|drop|disconnect|locked|unable|could not|couldn.t|not received",
    re.I)
# Trechos que NEGAM a queixa ("I never had any issues") são ignorados
NEGA_QUEIXA = re.compile(
    r"no (?:major |significant )?(?:issues|problems|complaints)|never (?:had|experienced|encountered)"
    r"|without (?:any )?(?:issues|problems)|not (?:had|experienced|encountered) any|haven.t had any"
    r"|have not had any|rarely|any issues|without any|hassle-free|free of",
    re.I)
SEM_EXIGIR_QUEIXA = {"concorrente", "pessoal", "sem_internet", "aumento_preco"}

CORTE = re.compile(r"(?<=[.!?;])\s+|,?\s+\b(?:but|however|while|although|though|whereas)\b", re.I)

def trechos(t):
    return [p.strip() for p in CORTE.split(t.strip()) if p and len(p.strip()) > 3]

def extrair_motivos(t):
    achados, evidencia = [], {}
    for p in trechos(t):
        queixa = bool(QUEIXA.search(p)) and not NEGA_QUEIXA.search(p)
        for m, padrao in MOTIVOS.items():
            if re.search(padrao, p, re.I) and (queixa or m in SEM_EXIGIR_QUEIXA):
                if m not in achados:
                    achados.append(m)          # guarda na ordem em que aparece no texto
                    evidencia[m] = p
    return achados, evidencia

def motivo_principal(t, achados):
    # Preferência: o motivo citado na mesma frase em que o cliente declara a saída
    for s in re.split(r"(?<=[.!?])\s+", t):
        if DECLAROU.search(s):
            for m in achados:
                if re.search(MOTIVOS[m], s, re.I):
                    return m
    return achados[0] if achados else "nao_identificado"

def processar(df):
    out = pd.DataFrame(index=df.index)
    out["customerID"] = df["customerID"]
    out["status_texto"] = df["CustomerFeedback"].map(status_texto)
    res = df["CustomerFeedback"].map(extrair_motivos)
    for m in MOTIVOS:
        out[f"motivo_{m}"] = res.map(lambda r: int(m in r[0]))
    out["motivos"] = res.map(lambda r: ", ".join(r[0]) or "nao_identificado")
    out["motivo_principal"] = [motivo_principal(t, r[0]) for t, r in zip(df["CustomerFeedback"], res)]
    out["trecho_evidencia"] = [r[1].get(p, "") for r, p in zip(res, out["motivo_principal"])]
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Preparação das features (só colunas tabulares)

# COMMAND ----------

CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
       "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
NUM = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "MonthlyIncome"]

def preparar(df):
    X = pd.DataFrame(index=df.index)
    for c in NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")        # "unknown" vira NaN (o modelo aceita NaN)
    for c in CAT:
        X[c] = df[c].astype(str).str.strip().str.title().astype("category")  # padroniza "yes"/"Yes "
    X["charges_por_renda"] = X["MonthlyCharges"] / X["MonthlyIncome"]       # peso da conta no bolso
    return X


def extrair_motivos(t):
    achados, evidencia = [], {}
    for p in trechos(t):
        queixa = bool(QUEIXA.search(p)) and not NEGA_QUEIXA.search(p)
        for m, padrao in MOTIVOS.items():
            if re.search(padrao, p, re.I) and (queixa or m in SEM_EXIGIR_QUEIXA):
                if m not in achados:
                    achados.append(m)          # guarda na ordem em que aparece no texto
                    evidencia[m] = p
    return achados, evidencia

def motivo_principal(t, achados):
    # Preferência: o motivo citado na mesma frase em que o cliente declara a saída
    for s in re.split(r"(?<=[.!?])\s+", t):
        if DECLAROU.search(s):
            for m in achados:
                if re.search(MOTIVOS[m], s, re.I):
                    return m
    return achados[0] if achados else "nao_identificado"

def processar(df):
    out = pd.DataFrame(index=df.index)
    out["customerID"] = df["customerID"]
    out["status_texto"] = df["CustomerFeedback"].map(status_texto)
    res = df["CustomerFeedback"].map(extrair_motivos)
    for m in MOTIVOS:
        out[f"motivo_{m}"] = res.map(lambda r: int(m in r[0]))
    out["motivos"] = res.map(lambda r: ", ".join(r[0]) or "nao_identificado")
    out["motivo_principal"] = [motivo_principal(t, r[0]) for t, r in zip(df["CustomerFeedback"], res)]
    out["trecho_evidencia"] = [r[1].get(p, "") for r, p in zip(res, out["motivo_principal"])]
    return out



X_hist = preparar(df_history)
X_inf = preparar(df_inference)
# garante as mesmas categorias nos dois
for c in CAT:
    cats = X_hist[c].cat.categories.union(X_inf[c].cat.categories)
    X_hist[c] = X_hist[c].cat.set_categories(cats)
    X_inf[c] = X_inf[c].cat.set_categories(cats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Rótulos de motivo, vindos do texto do HISTORY

# COMMAND ----------

mot = processar(df_history)                      # do notebook motivos_cancelamento
y_churn = (df_history["Churn"] == "Yes").astype(int)

ROTULOS = {
    "preco":       mot["motivo_preco"] | mot["motivo_aumento_preco"],
    "concorrente": mot["motivo_concorrente"],
    "pagamento":   mot["motivo_pagamento"],
    "tecnico":     mot["motivo_instabilidade"] | mot["motivo_velocidade"] | mot["motivo_sem_internet"],
}
churners = y_churn == 1
print("% dos churners com cada motivo:")
print({m: round(r[churners].mean() * 100, 1) for m, r in ROTULOS.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Treino com validação cruzada (previsões "fora da dobra")
# MAGIC Cada cliente recebe a previsão de um modelo que NÃO o viu no treino.
# MAGIC É isso que permite medir o desempenho de forma honesta.

# COMMAND ----------

cv = StratifiedKFold(5, shuffle=True, random_state=42)

def novo_modelo():
    return HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=300,
                                          learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=42)

# 3a) P(churn)
p_churn_oof = cross_val_predict(novo_modelo(), X_hist, y_churn, cv=cv, method="predict_proba")[:, 1]
print("AUC P(churn):", round(roc_auc_score(y_churn, p_churn_oof), 3))

# 3b) P(motivo | churn), treinado só em quem saiu
X_ch = X_hist[churners]
resultado = []
p_motivo_oof = {}
for m, r in ROTULOS.items():
    y_m = r[churners].astype(int)
    p = cross_val_predict(novo_modelo(), X_ch, y_m, cv=cv, method="predict_proba")[:, 1]
    p_motivo_oof[m] = p
    resultado.append({"motivo": m, "prevalencia_entre_churners": round(y_m.mean(), 3),
                      "AUC_P(m|churn)": round(roc_auc_score(y_m, p), 3)})
print(pd.DataFrame(resultado))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Quais características explicam cada motivo?
# MAGIC Importância por permutação: embaralha uma coluna e mede quanto o AUC cai.
# MAGIC Quanto mais cai, mais o modelo depende daquela coluna.

# COMMAND ----------

importancias = {}
for m, r in ROTULOS.items():
    y_m = r[churners].astype(int)
    mod = novo_modelo().fit(X_ch, y_m)
    imp = permutation_importance(mod, X_ch, y_m, scoring="roc_auc", n_repeats=5, random_state=42)
    importancias[m] = pd.Series(imp.importances_mean, index=X_ch.columns)
imp_df = pd.DataFrame(importancias).round(3)
for m in ROTULOS:
    print(f"\n{m}: top 4 colunas ->", imp_df[m].sort_values(ascending=False).head(4).to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Treino final em todo o HISTORY e escoragem do INFERENCE

# COMMAND ----------

mod_churn = novo_modelo().fit(X_hist, y_churn)
saida = pd.DataFrame({"customerID": df_inference["customerID"].values}, index=df_inference.index)
saida["p_churn"] = mod_churn.predict_proba(X_inf)[:, 1]

for m, r in ROTULOS.items():
    mod_m = novo_modelo().fit(X_ch, r[churners].astype(int))
    saida[f"p_motivo_{m}_se_sair"] = mod_m.predict_proba(X_inf)[:, 1]     # P(m | churn)
    saida[f"p_sair_por_{m}"] = saida["p_churn"] * saida[f"p_motivo_{m}_se_sair"]  # P(churn) x P(m|churn)

# Motivo que o modelo considera mais provável, e o que o cliente escreveu (quando escreveu)
cols_m = [f"p_motivo_{m}_se_sair" for m in ROTULOS]
saida["motivo_provavel_modelo"] = saida[cols_m].idxmax(axis=1).str.replace("p_motivo_", "").str.replace("_se_sair", "")
txt = processar(df_inference)
saida["status_texto"] = txt["status_texto"].values
saida["motivos_no_texto"] = txt["motivos"].values
saida["trecho_evidencia"] = txt["trecho_evidencia"].values

saida = saida.sort_values("p_churn", ascending=False)
display(saida) if "display" in globals() else print(saida.head(10).round(2).to_string())