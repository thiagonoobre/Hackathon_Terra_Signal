# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Validação dos modelos de motivo na base HISTÓRICA
# MAGIC Ordem de execução:
# MAGIC 1. `%pip install` não é necessário (só pandas e scikit-learn).
# MAGIC 2. Rode antes o notebook de regras: `%run ./motivos_cancelamento`
# MAGIC    (ele cria `processar`; o print final dele pode ser ignorado).
# MAGIC 3. Rode as células abaixo na ordem.

# COMMAND ----------

# CÉLULA 1: carregar a base histórica como pandas
# Ajuste o caminho. O multiLine é obrigatório (o feedback tem quebras de linha).
df_history = (spark.read.format("csv")
              .option("header", "true")
              .option("multiLine", "true")
              .option("escape", '"')
              .load("/Volumes/workspace/asn/metrics/history.csv")
              .toPandas())
print(df_history.shape)          # esperado: (5634, 23)

# COMMAND ----------

# CÉLULA 2: imports
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

# CÉLULA 3: gabarito (0/1 por motivo) a partir das regras
regras_hist = processar(df_history)
GABARITO = {
    "preco":       regras_hist["motivo_preco"] | regras_hist["motivo_aumento_preco"],
    "concorrente": regras_hist["motivo_concorrente"],
    "pagamento":   regras_hist["motivo_pagamento"],
    "tecnico":     regras_hist["motivo_instabilidade"] | regras_hist["motivo_velocidade"]
                   | regras_hist["motivo_sem_internet"],
}
print({m: int(g.sum()) for m, g in GABARITO.items()})   # esperado: preco 807, concorrente 574, pagamento 541, tecnico 872

# COMMAND ----------

# CÉLULA 4: o modelo (sem as palavras de decisão)
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

# CÉLULA 5: validação cruzada na base histórica inteira
def auc_seguro(y, p):
    # AUC precisa das duas classes (0 e 1). Se só existe uma, devolve NaN em vez de dar erro.
    return round(roc_auc_score(y, p), 3) if len(np.unique(y)) == 2 else np.nan

cv = StratifiedKFold(5, shuffle=True, random_state=42)
texto_hist = df_history["CustomerFeedback"].fillna("")

avaliacao, prob_oof = [], {}
for m, g in GABARITO.items():
    y = g.astype(int).values
    p = cross_val_predict(novo_modelo(), texto_hist, y, cv=cv, method="predict_proba")[:, 1]
    prob_oof[m] = p
    avaliacao.append({
        "motivo": m,
        "AUC": auc_seguro(y, p),
        "precisão p>0.5": round(precision_score(y, p > 0.5, zero_division=0), 2),
        "cobertura p>0.5": round(recall_score(y, p > 0.5, zero_division=0), 2),
        "casos na base": int(y.sum()),
    })
display(pd.DataFrame(avaliacao))

# COMMAND ----------

# CÉLULA 6: tabela de saída
saida_hist = pd.DataFrame({"customerID": df_history["customerID"].values})
for m in GABARITO:
    saida_hist[f"prob_{m}"] = prob_oof[m].round(3)

cols = [f"prob_{m}" for m in GABARITO]
saida_hist["motivo_mais_provavel"] = saida_hist[cols].idxmax(axis=1).str.replace("prob_", "")
saida_hist["prob_maxima"] = saida_hist[cols].max(axis=1)
saida_hist["status_texto"] = regras_hist["status_texto"].values
saida_hist["trecho_evidencia"] = regras_hist["trecho_evidencia"].values

# Visão de monitoramento: quem NÃO declarou saída, ordenado pela queixa mais forte
# monitorar_hist = (saida_hist[saida_hist["status_texto"] != "1_declarou_saida"]
#                   .sort_values("prob_maxima", ascending=False))
monitorar_hist = saida_hist
                  
display(monitorar_hist)

# COMMAND ----------


# spark.createDataFrame(monitorar_hist).write.mode("overwrite").saveAsTable("workspace.default.tbl_prop_feedback")
 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from  workspace.default.tbl_prop_feedback

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC with base as (
# MAGIC     select 
# MAGIC         TargetChurn,
# MAGIC         sum(case when prob_preco <= 0.7 then 0 else 1 end) as prob_preco,
# MAGIC         sum(case when prob_concorrente <= 0.7 then 0 else 1 end) as prob_concorrente,
# MAGIC         sum(case when prob_pagamento <= 0.7 then 0 else 1 end) as prob_pagamento,
# MAGIC         sum(case when prob_tecnico <= 0.7 then 0 else 1 end) as prob_tecnico
# MAGIC     from workspace.default.tbl_prop_feedback as a
# MAGIC     left join workspace.default.raw_abt2_churn as b
# MAGIC     on a.customerID = b.customerID
# MAGIC     group by 1 
# MAGIC     )
# MAGIC select 
# MAGIC     *
# MAGIC from base
# MAGIC