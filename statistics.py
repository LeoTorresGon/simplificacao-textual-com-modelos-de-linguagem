import json
import os
import pprint
import pandas as pd

path = r"D:\nilc_metrix\nilcmetrix\saidas"

files = []
if os.path.exists(path):
  folders = [
      f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
  ]
  for folder in folders:
    folder_path = os.path.join(path, folder)
    for file in os.listdir(folder_path):
      file_path = os.path.join(folder_path, file)
      if os.path.isfile(file_path):
        files.append(file_path)

modelos_validos = [
    "llama-3",
    "gpt-oss-120b",
    "qwen3-32b",
    "sabia1300_1500",
    "ministral",
    "corpus_split",
    "sabia_web",
    "gemini",
    "sabia_300-500",
]
prompts_validos = [
    "Identity,_Intent_and_Behaviour",
    "Tree_of_Thought",
    "Zero_Shot",
    "Original",
]
metricas_alvo = [
    "flesch",
    "words_per_sentence",
    "sentences_per_paragraph",
    "syllables_per_content_word",
    "cw_freq_brwac",
    "noun_ratio",
    "ttr",
    "gunning_fox",
    "brunet",
    "honore",
]

analise = {
    modelo: {prompt: {} for prompt in prompts_validos}
    for modelo in modelos_validos
}

# Lista para armazenar as observações individuais brutas
registros_brutos = []

for file_path in files:
  modelo_encontrado = None
  for modelo in modelos_validos:
    if modelo in file_path:
      modelo_encontrado = modelo
      break

  if not modelo_encontrado:
    continue

  if "Identity,_Intent_and_Behaviour" in file_path:
    prompt_encontrado = "Identity,_Intent_and_Behaviour"
  elif "Tree_of_Thought" in file_path:
    prompt_encontrado = "Tree_of_Thought"
  elif "corpus_split" in file_path:
    prompt_encontrado = "Original"
  else:
    prompt_encontrado = "Zero_Shot"

  try:
    with open(file_path, "r", encoding="utf-8") as f:
      metricas_arquivo = json.load(f)

    contexto_prompt = analise[modelo_encontrado][prompt_encontrado]
    nome_arquivo = os.path.basename(file_path)

    for metrica, valor in metricas_arquivo.items():
      if valor == 0 or valor == 0.0:
        continue

      # Registra para o cálculo das médias
      if metrica not in contexto_prompt:
        contexto_prompt[metrica] = {"soma": 0.0, "quantidade_valida": 0}

      val_float = float(valor)
      contexto_prompt[metrica]["soma"] += val_float
      contexto_prompt[metrica]["quantidade_valida"] += 1

      # Registra o ponto individual bruto para o Violin Plot
      if metrica in metricas_alvo:
        registros_brutos.append({
            "modelo": modelo_encontrado,
            "prompt": prompt_encontrado,
            "metrica": metrica,
            "valor": val_float,
            "arquivo": nome_arquivo,
        })

  except Exception as e:
    print(f"Erro ao processar o arquivo {file_path}: {e}")

# ---------------------------------------------------------------------------
# 1. EXPORTAÇÃO DOS DADOS BRUTOS (PARA VIOLIN PLOT E BOXPLOT)
# ---------------------------------------------------------------------------
df_bruto = pd.DataFrame(registros_brutos)
df_bruto.to_csv(
    "dados_brutos_para_violinplot.csv", index=False, encoding="utf-8-sig"
)

with open(
    "dados_brutos_para_violinplot.json", "w", encoding="utf-8"
) as f_bruto_json:
  json.dump(registros_brutos, f_bruto_json, ensure_ascii=False, indent=4)

print("Arquivo 'dados_brutos_para_violinplot.csv' gerado com sucesso!")

# ---------------------------------------------------------------------------
# 2. MANUTENÇÃO DOS CÁLCULOS DE MÉDIAS CONSOLIDADAS EXISTENTES
# ---------------------------------------------------------------------------
resultados_finais = {
    modelo: {prompt: {} for prompt in prompts_validos}
    for modelo in modelos_validos
}

for modelo in analise:
  for prompt in analise[modelo]:
    for metrica, dados in analise[modelo][prompt].items():
      soma = dados["soma"]
      qtd = dados["quantidade_valida"]

      if qtd > 0:
        resultados_finais[modelo][prompt][metrica] = round(soma / qtd, 5)
        if qtd != 10:
          resultados_finais[modelo][prompt][f"{metrica}_count"] = qtd

with open("analise_consolidada_modelos.json", "w", encoding="utf-8") as f:
  json.dump(resultados_finais, f, ensure_ascii=False, indent=4)

resultados_finais_filtrados = {}
for mod, prompts in resultados_finais.items():
  resultados_finais_filtrados[mod] = {}
  for prm, metrs in prompts.items():
    resultados_finais_filtrados[mod][prm] = {
        m: v
        for m, v in metrs.items()
        if m in metricas_alvo or m.replace("_count", "") in metricas_alvo
    }

with open(
    "analise_consolidada_modelos_filtrada.json", "w", encoding="utf-8"
) as f:
  json.dump(resultados_finais_filtrados, f, ensure_ascii=False, indent=4)


def media_modelo():
  medias_modelo = {}

  for modelo in resultados_finais:
    medias_modelo[modelo] = {}
    contagem_prompts = {}

    for prompt in resultados_finais[modelo]:
      for metrica in resultados_finais[modelo][prompt]:
        if metrica.endswith("_count"):
          continue

        if metrica not in medias_modelo[modelo]:
          medias_modelo[modelo][metrica] = 0.0
          contagem_prompts[metrica] = 0

        medias_modelo[modelo][metrica] += resultados_finais[modelo][prompt][
            metrica
        ]
        contagem_prompts[metrica] += 1

    for metrica in medias_modelo[modelo]:
      qtd_prompts = contagem_prompts[metrica]
      if qtd_prompts > 0:
        medias_modelo[modelo][metrica] = round(
            medias_modelo[modelo][metrica] / qtd_prompts, 5
        )

  with open("medias_consolidadas_por_modelo.json", "w", encoding="utf-8") as f:
    json.dump(medias_modelo, f, ensure_ascii=False, indent=4)

  medias_modelo_filtrada = {
      mod: {m: v for m, v in metrs.items() if m in metricas_alvo}
      for mod, metrs in medias_modelo.items()
  }
  with open(
      "medias_consolidadas_por_modelo_filtrada.json", "w", encoding="utf-8"
  ) as f:
    json.dump(medias_modelo_filtrada, f, ensure_ascii=False, indent=4)

  print("Métricas consolidadas por modelo geradas com sucesso!")


def media_prompt():
  medias_prompt = {}

  for modelo in resultados_finais:
    for prompt in resultados_finais[modelo]:
      if prompt not in medias_prompt:
        medias_prompt[prompt] = {}
        medias_prompt[prompt]["_contagem_modelos"] = {}

      contexto_prompt = medias_prompt[prompt]

      for metrica in resultados_finais[modelo][prompt]:
        if metrica.endswith("_count"):
          continue

        if metrica not in contexto_prompt:
          contexto_prompt[metrica] = 0.0
          contexto_prompt["_contagem_modelos"][metrica] = 0

        contexto_prompt[metrica] += resultados_finais[modelo][prompt][metrica]
        contexto_prompt["_contagem_modelos"][metrica] += 1

  for prompt in medias_prompt:
    contagem_modelos = medias_prompt[prompt].pop("_contagem_modelos")

    for metrica in medias_prompt[prompt]:
      qtd_modelos = contagem_modelos[metrica]
      if qtd_modelos > 0:
        medias_prompt[prompt][metrica] = round(
            medias_prompt[prompt][metrica] / qtd_modelos, 5
        )

  with open("medias_consolidadas_por_prompt.json", "w", encoding="utf-8") as f:
    json.dump(medias_prompt, f, ensure_ascii=False, indent=4)

  medias_prompt_filtrada = {
      prm: {m: v for m, v in metrs.items() if m in metricas_alvo}
      for prm, metrs in medias_prompt.items()
  }
  with open(
      "medias_consolidadas_por_prompt_filtrada.json", "w", encoding="utf-8"
  ) as f:
    json.dump(medias_prompt_filtrada, f, ensure_ascii=False, indent=4)

  print("Métricas consolidadas por prompt geradas com sucesso!")


def media_prompt_sem_sabia():
  medias_prompt = {}

  for modelo in resultados_finais:
    if modelo == "sabia1300_1500" or modelo == "sabia_300-500":
      continue
    for prompt in resultados_finais[modelo]:
      if prompt not in medias_prompt:
        medias_prompt[prompt] = {}
        medias_prompt[prompt]["_contagem_modelos"] = {}

      contexto_prompt = medias_prompt[prompt]

      for metrica in resultados_finais[modelo][prompt]:
        if metrica.endswith("_count"):
          continue

        if metrica not in contexto_prompt:
          contexto_prompt[metrica] = 0.0
          contexto_prompt["_contagem_modelos"][metrica] = 0

        contexto_prompt[metrica] += resultados_finais[modelo][prompt][metrica]
        contexto_prompt["_contagem_modelos"][metrica] += 1

  for prompt in medias_prompt:
    contagem_modelos = medias_prompt[prompt].pop("_contagem_modelos")

    for metrica in medias_prompt[prompt]:
      qtd_modelos = contagem_modelos[metrica]
      if qtd_modelos > 0:
        medias_prompt[prompt][metrica] = round(
            medias_prompt[prompt][metrica] / qtd_modelos, 5
        )

  with open(
      "medias_consolidadas_por_prompt_sem_sabia.json", "w", encoding="utf-8"
  ) as f:
    json.dump(medias_prompt, f, ensure_ascii=False, indent=4)

  medias_prompt_filtrada_sem_sabia = {
      prm: {m: v for m, v in metrs.items() if m in metricas_alvo}
      for prm, metrs in medias_prompt.items()
  }
  with open(
      "medias_consolidadas_por_prompt_sem_sabia_filtrada.json",
      "w",
      encoding="utf-8",
  ) as f:
    json.dump(
        medias_prompt_filtrada_sem_sabia, f, ensure_ascii=False, indent=4
    )

  print("Métricas consolidadas por prompt sem Sabiá geradas com sucesso!")


media_prompt()
media_modelo()
media_prompt_sem_sabia()