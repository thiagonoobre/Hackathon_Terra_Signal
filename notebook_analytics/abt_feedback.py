# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "vaderSentiment",
#   "textblob",
# ]
# ///
# MAGIC %md
# MAGIC # Features de CustomerFeedback: TF-IDF + SVD + Sentimento
# MAGIC Pré-requisito: `%pip install vaderSentiment textblob` e um `df` pandas com as colunas
# MAGIC `CustomerFeedback` e `Churn` (ex.: `df = spark_df.toPandas()`).

# COMMAND ----------

# MAGIC %pip install vaderSentiment textblob

# COMMAND ----------


df = (spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiLine", "true")   # permite quebras de linha dentro de aspas
      .option("escape", '"')         # trata "" dentro do texto como aspas literais
      .load("/Volumes/workspace/asn/metrics/history.csv"))
      

df = df.toPandas()

# COMMAND ----------

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

y = (df["Churn"] == "Yes").astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) TF-IDF  (uma coluna por palavra)

# COMMAND ----------

v = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_df=0.8,
                    stop_words="english", max_features=100)
X = v.fit_transform(df["CustomerFeedback"])
palavras = v.get_feature_names_out()
print("matriz TF-IDF:", X.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) SVD  (comprime as 100 colunas em k componentes)

# COMMAND ----------

K = 5
svd = TruncatedSVD(n_components=K, random_state=42)
Z = svd.fit_transform(X)

# 2a) AS COLUNAS NOVAS (uma linha por cliente)
svd_df = pd.DataFrame(Z, columns=[f"svd_{i+1}" for i in range(K)], index=df.index)
display(svd_df.head()) if "display" in globals() else print(svd_df.head().round(2))

# 2b) OS PESOS: quanto cada palavra pesa em cada componente (K linhas x 100 palavras)
pesos = pd.DataFrame(svd.components_, index=svd_df.columns, columns=palavras)

# 2c) Como ler UM componente: palavras que puxam para cada lado
def ler_componente(nome, n=6):
    s = pesos.loc[nome].sort_values()
    print(f"\n{nome}\n  puxam para o lado NEGATIVO:", s.head(n).round(2).to_dict())
    print(f"  puxam para o lado POSITIVO:", s.tail(n).round(2).to_dict())

for c in svd_df.columns:
    ler_componente(c)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Relação de cada componente com o churn

# COMMAND ----------

# 3a) Média de cada componente em quem saiu x quem ficou
medias = svd_df.groupby(df["Churn"]).mean().round(3)
print(medias)

# 3b) Correlação de cada componente com o churn (0 = nenhuma, 1 = perfeita)
print(svd_df.corrwith(y).round(2))

# 3c) Variância explicada: quanto da informação original os K componentes mantêm
print("variância explicada (soma):", svd.explained_variance_ratio_.sum().round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Conferindo a conta de UM cliente (valor do svd = soma de TF-IDF x peso)

# COMMAND ----------

def explicar_cliente(i, comp="svd_2"):
    tfidf_i = pd.Series(X[i].toarray().ravel(), index=palavras)
    contrib = (tfidf_i * pesos.loc[comp]).loc[lambda s: s != 0].sort_values()
    print(f"Cliente {i} | Churn = {df['Churn'].iloc[i]} | {comp} = {svd_df.loc[i, comp]:.3f}")
    print("palavras que mais empurram para cima:", contrib.tail(5).round(3).to_dict())
    print("palavras que mais empurram para baixo:", contrib.head(3).round(3).to_dict())
    print("soma das contribuições:", round(contrib.sum(), 3), "(deve ser igual ao valor do svd)")

explicar_cliente(1, "svd_2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Sentimento (3 colunas) + features para complementar

# COMMAND ----------

vader = SentimentIntensityAnalyzer()

def frases(texto):
    return [s for s in re.split(r"(?<=[.!?])\s+", texto.strip()) if len(s) > 3]

sent = pd.DataFrame(index=df.index)
txt = df["CustomerFeedback"]

# As três colunas principais
sent["vader_compound"]     = txt.map(lambda t: vader.polarity_scores(t)["compound"])
sent["textblob_polarity"]  = txt.map(lambda t: TextBlob(t).sentiment.polarity)
sent["textblob_subjectivity"] = txt.map(lambda t: TextBlob(t).sentiment.subjectivity)

# Complementares: olham a ESTRUTURA do texto, não só o tom médio
sent["vader_primeira_frase"] = txt.map(lambda t: vader.polarity_scores(frases(t)[0])["compound"])
sent["tem_contraste"] = txt.str.contains(r"\b(?:but|however|although|though)\b",
                                         case=False).astype(int)

sent["vader_faixa"] = pd.cut(sent["vader_compound"], [-1.01, -0.05, 0.05, 1.01],
                             labels=["negativo", "neutro", "positivo"])
sent.head()

# COMMAND ----------

# Churn por faixa de sentimento
print(pd.crosstab(sent["vader_faixa"], df["Churn"], normalize="index").round(2))

# Dentro dos textos POSITIVOS: o que separa quem saiu de quem ficou?
pos = sent["vader_faixa"] == "positivo"
print("\nChurn dentro dos positivos, por contraste (but/however...):")
print(pd.crosstab(sent.loc[pos, "tem_contraste"], df.loc[pos, "Churn"], normalize="index").round(2))
print("\nMédia da 1ª frase dentro dos positivos:")
print(sent.loc[pos].groupby(df.loc[pos, "Churn"])["vader_primeira_frase"].mean().round(2))