# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Camada de apoio ao atendente: intenção declarada + motivos de cancelamento
# MAGIC **Não entra no modelo de escoragem.** Lê o `CustomerFeedback` e devolve, para cada cliente:
# MAGIC - `status_texto`: o que o próprio cliente disse (declarou saída / insatisfeito / satisfeito)
# MAGIC - `motivo_*`: uma coluna 0/1 por motivo de RECLAMAÇÃO (não basta citar o assunto)
# MAGIC - `motivos` e `motivo_principal`: o resumo para o atendente
# MAGIC - `trecho_evidencia`: a frase do cliente que sustenta o motivo principal
# MAGIC
# MAGIC Só usa pandas e expressões regulares (nenhuma instalação extra).

# COMMAND ----------

import re
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) O cliente DECLAROU que vai sair?

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

# MAGIC %md
# MAGIC ## 2) Motivos: o assunto aparece EM TOM DE RECLAMAÇÃO?
# MAGIC Cada texto é quebrado em **trechos**: primeiro em frases, depois em cada "but", "while",
# MAGIC "however", "although", ";". Um motivo só conta se o MESMO trecho:
# MAGIC 1. fala do assunto (palavras do tema), **e**
# MAGIC 2. tem palavra de queixa ("too high", "slow", "outages", "inconvenient"...).
# MAGIC
# MAGIC Exemplo: "While the fiber was fast and reliable, | I found the charges too high"
# MAGIC vira 2 trechos. Só o segundo tem queixa, então o motivo é **preço**, e não instabilidade.

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
# MAGIC ## 3) Validação no HISTORY (onde sabemos o Churn real)

# COMMAND ----------


df_history = (spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiLine", "true")   # permite quebras de linha dentro de aspas
      .option("escape", '"')         # trata "" dentro do texto como aspas literais
      .load("/Volumes/workspace/asn/metrics/history.csv"))



df_inference = (spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiLine", "true")   # permite quebras de linha dentro de aspas
      .option("escape", '"')         # trata "" dentro do texto como aspas literais
      .load("/Volumes/workspace/asn/metrics/inference.csv"))
      

# COMMAND ----------

df_hist = df_history.toPandas()  # processar trabalha com pandas; df_history é Spark
hist_mot = processar(df_hist)
print(pd.crosstab(hist_mot["status_texto"], df_hist["Churn"], margins=True))

declarou = hist_mot["status_texto"] == "1_declarou_saida"
churn = df_hist["Churn"] == "Yes"
print("Precisão  (dos que o texto marcou, quantos saíram):", round(churn[declarou].mean(), 3))
print("Cobertura (dos que saíram, quantos o texto marcou):", round(declarou[churn].mean(), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Aplicação no INFERENCE

# COMMAND ----------

df_infer = df_inference.toPandas() 

inf_mot = processar(df_infer)
print(inf_mot["status_texto"].value_counts())
display(inf_mot) if "display" in globals() else print(inf_mot.head())