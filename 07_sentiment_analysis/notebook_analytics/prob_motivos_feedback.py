# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Probabilidade de cada motivo de insatisfação, a partir do CustomerFeedback
# MAGIC **Objetivo:** monitorar clientes (principalmente os que NÃO saíram) e dizer ao operador,
# MAGIC para cada um, a probabilidade de o feedback conter uma queixa de:
# MAGIC **preço · concorrente · forma de pagamento · técnico (velocidade/instabilidade)**.
# MAGIC
# MAGIC Não usa Churn em nenhum momento.
# MAGIC
# MAGIC **Como funciona:**
# MAGIC 1. As regras do notebook `motivos_cancelamento` marcam 0/1 em cada texto do HISTORY (o "gabarito").
# MAGIC 2. Para cada motivo, um TF-IDF + Regressão Logística aprende com esse gabarito.
# MAGIC 3. O modelo devolve uma PROBABILIDADE (0 a 1) em vez de só 0/1, e generaliza para frases
# MAGIC    que as regras não previam.
# MAGIC
# MAGIC Pré-requisito: `%run ./motivos_cancelamento` (função `processar`).

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import re
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Gabarito: o que as regras encontram em cada texto

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

regras_hist = processar(df_history)
GABARITO = {
    "preco":       regras_hist["motivo_preco"] | regras_hist["motivo_aumento_preco"],
    "concorrente": regras_hist["motivo_concorrente"],
    "pagamento":   regras_hist["motivo_pagamento"],
    "tecnico":     regras_hist["motivo_instabilidade"] | regras_hist["motivo_velocidade"]
                   | regras_hist["motivo_sem_internet"],
}
print(pd.DataFrame({m: [g.sum(), round(g.mean() * 100, 1)] for m, g in GABARITO.items()},
                   index=["textos com o motivo", "%"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) O modelo de cada motivo
# MAGIC Removemos as palavras de DECISÃO de sair (decided, churn, cancel, switch...).
# MAGIC Assim o modelo aprende a reconhecer a **queixa** ("charges too high"), e não a decisão.
# MAGIC Isso é o que permite usar o modelo em quem NÃO saiu.

# COMMAND ----------

PALAVRAS_DE_DECISAO = ["decided", "decide", "deciding", "decision", "churn", "churned", "churning",
                       "cancel", "cancelled", "canceled", "canceling", "cancelling", "switch",
                       "switched", "switching", "leave", "leaving", "left", "ultimately",
                       "unfortunately", "former"]
STOP = list(TfidfVectorizer(stop_words="english").get_stop_words()) + PALAVRAS_DE_DECISAO

def novo_modelo():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), min_df=5, stop_words=STOP, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4, class_weight="balanced"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Validação (previsões fora da dobra)
# MAGIC Medimos o desempenho em todos e, separadamente, **só em quem não saiu** (o público-alvo).
# MAGIC O Churn aqui serve APENAS para separar esse grupo na avaliação; ele não entra no modelo.

# COMMAND ----------

cv = StratifiedKFold(5, shuffle=True, random_state=42)
ficou = (df_history["Churn"] == "No").values
texto_hist = df_history["CustomerFeedback"]

avaliacao, prob_oof = [], {}
for m, g in GABARITO.items():
    y = g.astype(int).values
    p = cross_val_predict(novo_modelo(), texto_hist, y, cv=cv, method="predict_proba")[:, 1]
    prob_oof[m] = p
    avaliacao.append({
        "motivo": m,
        "AUC (todos)": round(roc_auc_score(y, p), 3),
        "AUC (só quem ficou)": round(roc_auc_score(y[ficou], p[ficou]), 3),
        "precisão p>0.5 (quem ficou)": round(precision_score(y[ficou], p[ficou] > 0.5, zero_division=0), 2),
        "cobertura p>0.5 (quem ficou)": round(recall_score(y[ficou], p[ficou] > 0.5), 2),
        "casos em quem ficou": int(y[ficou].sum()),
    })
print(pd.DataFrame(avaliacao).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) O que o modelo aprendeu: palavras que mais aumentam cada probabilidade

# COMMAND ----------

modelos = {}
for m, g in GABARITO.items():
    modelos[m] = novo_modelo().fit(texto_hist, g.astype(int))
    tfidf, lr = modelos[m].named_steps.values()
    termos = np.array(tfidf.get_feature_names_out())
    print(f"{m:12s} ->", list(termos[np.argsort(lr.coef_[0])[-10:][::-1]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Escoragem do INFERENCE

# COMMAND ----------

texto_inf = df_inference["CustomerFeedback"]
regras_inf = processar(df_inference)

saida = pd.DataFrame({"customerID": df_inference["customerID"].values})
for m, mod in modelos.items():
    saida[f"prob_{m}"] = mod.predict_proba(texto_inf)[:, 1].round(3)

cols = [f"prob_{m}" for m in modelos]
saida["motivo_mais_provavel"] = saida[cols].idxmax(axis=1).str.replace("prob_", "")
saida["prob_maxima"] = saida[cols].max(axis=1)
saida["status_texto"] = regras_inf["status_texto"].values      # declarou saída / insatisfeito / satisfeito
saida["trecho_evidencia"] = regras_inf["trecho_evidencia"].values

# Visão de monitoramento: quem NÃO declarou saída, ordenado pela maior queixa
monitorar = saida #saida[saida["status_texto"] != "1_declarou_saida"].sort_values("prob_maxima", ascending=False)
display(monitorar) if "display" in globals() else print(monitorar.head(10).to_string())

# COMMAND ----------

# DBTITLE 1,Monitorar (SQL)

spark.createDataFrame(monitorar).write.mode("overwrite").saveAsTable("workspace.default.tbl_prop_feedback")
 

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC     *
# MAGIC from workspace.default.tbl_prop_feedback as a
# MAGIC left join workspace.default.raw_abt2_churn as b
# MAGIC    on a.customerID = b.customerID
# MAGIC