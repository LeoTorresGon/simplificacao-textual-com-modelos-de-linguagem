import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_PATH = "relatorio_estatistico_ic.csv"

# Modelos que participam do ranking principal
MODELOS_VALIDOS = [
    "ministral",
    "llama-3",
    "qwen3-32b",
    "gpt-oss-120b",
    "gemini",
]

# Estratégias utilizadas no ranking principal
PROMPTS_VALIDOS = [
    "Zero_Shot",
    "Identity,_Intent_and_Behaviour",
    "Tree_of_Thought",
]

# Critérios utilizados no TOPSIS
CRITERIOS = [
    "flesch",
    "words_per_sentence",
    "syllables_per_content_word",
    "cw_freq_brwac",
    "gunning_fox",
]

# Orientação:
# True  = maior é melhor (benefício)
# False = menor é melhor (custo)
#
# ATENÇÃO:
# Estas orientações precisam corresponder ao que você definiu
# metodologicamente no SAW.
BENEFICIO = {
    "flesch": True,
    "words_per_sentence": False,
    "syllables_per_content_word": False,
    "cw_freq_brwac": True,
    "gunning_fox": False,
}

pesos = {
    "flesch": 0.35,  # Legibilidade Global
    "words_per_sentence": 0.25,  # Comprimento de Frase (Sintaxe)
    "gunning_fox": 0.20,  # Nível de Escolaridade Exigido
    "cw_freq_brwac": 0.15,  # Popularidade Lexical (Vocabulário Cotidiano)
    "syllables_per_content_word": 0.05,  # Extensão de Palavras (Morfologia)
}

PESOS = {
    "flesch": 0.35,
    "words_per_sentence": 0.25,
    "syllables_per_content_word": 0.05,
    "cw_freq_brwac": 0.15,
    "gunning_fox": 0.20,
}


# ============================================================
# VALIDAÇÕES
# ============================================================

def validar_configuracao():
    if set(CRITERIOS) != set(PESOS.keys()):
        raise ValueError("Os critérios e os pesos não coincidem.")

    if set(CRITERIOS) != set(BENEFICIO.keys()):
        raise ValueError("Os critérios e as orientações não coincidem.")

    soma_pesos = sum(PESOS.values())

    if not np.isclose(soma_pesos, 1.0):
        raise ValueError(
            f"A soma dos pesos deve ser 1.0. Soma atual: {soma_pesos}"
        )


# ============================================================
# TOPSIS
# ============================================================

def topsis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o método TOPSIS.

    Parâmetros
    ----------
    df : DataFrame
        Linhas = alternativas
        Colunas = critérios

    Retorno
    -------
    DataFrame
        Ranking TOPSIS com distâncias e coeficiente de proximidade.
    """

    matriz = df[CRITERIOS].astype(float).to_numpy()

    # --------------------------------------------------------
    # 1. Normalização vetorial
    # --------------------------------------------------------
    denominadores = np.sqrt((matriz ** 2).sum(axis=0))

    if np.any(denominadores == 0):
        raise ValueError(
            "Existe pelo menos um critério com todas as observações iguais a zero."
        )

    matriz_normalizada = matriz / denominadores

    # --------------------------------------------------------
    # 2. Aplicação dos pesos
    # --------------------------------------------------------
    pesos = np.array([PESOS[c] for c in CRITERIOS])

    matriz_ponderada = matriz_normalizada * pesos

    # --------------------------------------------------------
    # 3. Solução ideal positiva e negativa
    # --------------------------------------------------------
    ideal_positivo = np.zeros(len(CRITERIOS))
    ideal_negativo = np.zeros(len(CRITERIOS))

    for j, criterio in enumerate(CRITERIOS):

        coluna = matriz_ponderada[:, j]

        if BENEFICIO[criterio]:
            # Maior é melhor
            ideal_positivo[j] = coluna.max()
            ideal_negativo[j] = coluna.min()

        else:
            # Menor é melhor
            ideal_positivo[j] = coluna.min()
            ideal_negativo[j] = coluna.max()

    # --------------------------------------------------------
    # 4. Distância para o ideal positivo e negativo
    # --------------------------------------------------------
    distancia_ideal_positivo = np.sqrt(
        ((matriz_ponderada - ideal_positivo) ** 2).sum(axis=1)
    )

    distancia_ideal_negativo = np.sqrt(
        ((matriz_ponderada - ideal_negativo) ** 2).sum(axis=1)
    )

    # --------------------------------------------------------
    # 5. Coeficiente de proximidade relativa
    # --------------------------------------------------------
    denominador = (
        distancia_ideal_positivo + distancia_ideal_negativo
    )

    proximidade = np.divide(
        distancia_ideal_negativo,
        denominador,
        out=np.zeros_like(denominador),
        where=denominador != 0
    )

    # --------------------------------------------------------
    # 6. Resultado
    # --------------------------------------------------------
    resultado = df[
        ["modelo", "prompt"]
    ].copy()

    resultado["distancia_ideal_positivo"] = distancia_ideal_positivo
    resultado["distancia_ideal_negativo"] = distancia_ideal_negativo
    resultado["coeficiente_topsis"] = proximidade

    resultado = resultado.sort_values(
        by="coeficiente_topsis",
        ascending=False
    ).reset_index(drop=True)

    resultado["posicao"] = resultado.index + 1

    return resultado[
        [
            "posicao",
            "modelo",
            "prompt",
            "coeficiente_topsis",
            "distancia_ideal_positivo",
            "distancia_ideal_negativo",
        ]
    ]


# ============================================================
# LEITURA E PREPARAÇÃO DOS DADOS
# ============================================================

def main():

    validar_configuracao()

    # --------------------------------------------------------
    # Leitura do CSV
    # --------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    # --------------------------------------------------------
    # Filtra somente as alternativas usadas no ranking principal
    # --------------------------------------------------------
    df = df[
        df["modelo"].isin(MODELOS_VALIDOS)
        & df["prompt"].isin(PROMPTS_VALIDOS)
    ].copy()

    # --------------------------------------------------------
    # Usa a média de cada métrica como valor do critério
    #
    # Cada combinação modelo + prompt possui 10 métricas.
    # --------------------------------------------------------
    matriz = (
        df.pivot_table(
            index=["modelo", "prompt"],
            columns="metrica",
            values="mean",
            aggfunc="mean"
        )
        .reset_index()
    )

    # Verificação
    colunas_necessarias = ["modelo", "prompt"] + CRITERIOS

    faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in matriz.columns
    ]

    if faltantes:
        raise ValueError(
            f"Critérios ausentes no CSV: {faltantes}"
        )

    matriz = matriz[colunas_necessarias]

    # --------------------------------------------------------
    # TOPSIS
    # --------------------------------------------------------
    ranking = topsis(matriz)

    # --------------------------------------------------------
    # Apresentação
    # --------------------------------------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== RANKING TOPSIS ===\n")

    print(
        ranking.to_string(
            index=False,
            formatters={
                "coeficiente_topsis": "{:.4f}".format,
                "distancia_ideal_positivo": "{:.4f}".format,
                "distancia_ideal_negativo": "{:.4f}".format,
            }
        )
    )

    # --------------------------------------------------------
    # Exporta resultado
    # --------------------------------------------------------
    ranking.to_csv(
        "ranking_topsis.csv",
        index=False
    )

    print("\nArquivo salvo: ranking_topsis.csv")


if __name__ == "__main__":
    main()