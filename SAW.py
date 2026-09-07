import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CARREGAMENTO E FILTRAGEM DOS DADOS
# ---------------------------------------------------------------------------
csv_filename = "relatorio_estatistico_ic.csv"
df = pd.read_csv(csv_filename)

# Exclusão das variantes Sabiá
sabias = ["sabia1300_1500", "sabia_300-500", "sabia_web"]
df_filtered = df[~df["modelo"].isin(sabias)].copy()

model_clean = {
    "ministral": "Ministral 3 (3B)",
    "gemini": "Gemini Flash 3.5",
    "qwen3-32b": "Qwen3-32B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "llama-3": "Llama 3.1 (8B)",
    "corpus_split": "Original",
}

prompt_clean = {
    "Zero_Shot": "Zero-Shot",
    "Identity,_Intent_and_Behaviour": "IIB",
    "Tree_of_Thought": "ToT",
    "Original": "Original",
}

# Criar pivô com as médias observadas para cada alternativa (Modelo + Prompt)
df_pivot = df_filtered.pivot_table(
    index=["modelo", "prompt"], columns="metrica", values="mean"
).reset_index()

# Remover o Texto Original para ranquear apenas as alternativas de IA
df_pivot = df_pivot[df_pivot["modelo"] != "corpus_split"].copy()

# Mapear rótulos limpos
df_pivot["Modelo"] = df_pivot["modelo"].map(model_clean)
df_pivot["Prompt"] = df_pivot["prompt"].map(prompt_clean)

# ---------------------------------------------------------------------------
# 2. DEFINIÇÃO DOS CRITÉRIOS E PESOS DO MÉTODO SAW
# ---------------------------------------------------------------------------
# Orientações:
criterios_beneficio = ["flesch", "cw_freq_brwac"]  # Maior é melhor ↑
criterios_custo = [
    "words_per_sentence",
    "gunning_fox",
    "syllables_per_content_word",
]  # Menor é melhor ↓

# Vetor de Pesos Ponderados (Soma = 1.0 / 100%)
pesos = {
    "flesch": 0.35,  # Legibilidade Global
    "words_per_sentence": 0.25,  # Comprimento de Frase (Sintaxe)
    "gunning_fox": 0.20,  # Nível de Escolaridade Exigido
    "cw_freq_brwac": 0.15,  # Popularidade Lexical (Vocabulário Cotidiano)
    "syllables_per_content_word": 0.05,  # Extensão de Palavras (Morfologia)
}

# ---------------------------------------------------------------------------
# 3. NORMALIZAÇÃO MIN-MAX [0, 1] E CÁLCULO DA PONTUAÇÃO GLOBAL
# ---------------------------------------------------------------------------
norm_df = pd.DataFrame(index=df_pivot.index)

# Normalização de Benefício (Maior = Melhor)
for col in criterios_beneficio:
  min_v = df_pivot[col].min()
  max_v = df_pivot[col].max()
  norm_df[col] = (df_pivot[col] - min_v) / (max_v - min_v)

# Normalização de Custo (Menor = Melhor)
for col in criterios_custo:
  min_v = df_pivot[col].min()
  max_v = df_pivot[col].max()
  norm_df[col] = (max_v - df_pivot[col]) / (max_v - min_v)

# Cálculo do Score SAW (Soma Ponderada)
df_pivot["Score_SAW"] = 0.0
for metric, peso in pesos.items():
  df_pivot["Score_SAW"] += norm_df[metric] * peso

# ---------------------------------------------------------------------------
# 4. ORDENAÇÃO E MONTAGEM DA TABELA FINAL
# ---------------------------------------------------------------------------
df_ranking = df_pivot.sort_values(
    by="Score_SAW", ascending=False
).reset_index(drop=True)
df_ranking["Posição"] = df_ranking.index + 1

# Arredondamento para formato legível
cols_mcdm = [
    "Posição",
    "Modelo",
    "Prompt",
    "Score_SAW",
    "flesch",
    "words_per_sentence",
    "gunning_fox",
    "cw_freq_brwac",
    "syllables_per_content_word",
]
tabela_final = df_ranking[cols_mcdm].copy()

tabela_final["Score_SAW"] = tabela_final["Score_SAW"].round(4)
tabela_final["flesch"] = tabela_final["flesch"].round(2)
tabela_final["words_per_sentence"] = tabela_final["words_per_sentence"].round(
    2
)
tabela_final["gunning_fox"] = tabela_final["gunning_fox"].round(2)
tabela_final["cw_freq_brwac"] = tabela_final["cw_freq_brwac"].round(2)
tabela_final["syllables_per_content_word"] = tabela_final[
    "syllables_per_content_word"
].round(2)

# Exibição no Console
print("\n" + "=" * 90)
print("RANKING FINAL MULTICRITÉRIO (MÉTODO SAW)")
print("=" * 90)
print(tabela_final.to_string(index=False))
print("=" * 90)

# Exportar para CSV
tabela_final.to_csv("ranking_multicriterio_saw.csv", index=False)

# Exportar direto para tabela LaTeX
latex_code = tabela_final.to_latex(
    index=False,
    caption=(
        "Ranking final das alternativas de simplificação pelo Método Multicritério"
        " SAW."
    ),
    label="tab:ranking_saw",
    column_format="c|cc|c|ccccc",
    position="htbp",
)

with open("tabela_ranking_saw.tex", "w", encoding="utf-8") as f:
  f.write(latex_code)

print(
    "\n[SUCESSO] Tabela gerada e salva em 'ranking_multicriterio_saw.csv' e"
    " 'tabela_ranking_saw.tex'!"
)