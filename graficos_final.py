import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# 1. ESTILO E CARREGAMENTO DOS DADOS
# ---------------------------------------------------------------------------
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8.5

# Lista padronizada de modelos a excluir
sabias = ["sabia1300_1500", "sabia_300-500", "sabia_web", "corpus_split"]

df = pd.read_csv("relatorio_estatistico_ic.csv")

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

colors_dict = {
    "Ministral 3 (3B)": "#1f77b4",  # Azul
    "Gemini Flash 3.5": "#ff7f0e",  # Laranja
    "Qwen3-32B": "#2ca02c",  # Verde
    "GPT-OSS 120B": "#d62728",  # Vermelho
    "Llama 3.1 (8B)": "#9467bd",  # Roxo
    "Original": "#555555",
}

marker_dict = {
    "Zero_Shot": "o",
    "Identity,_Intent_and_Behaviour": "s",  # Quadrado = IIB
    "Tree_of_Thought": "^",  # Triângulo = ToT
    "Original": "X",
}

modelos_keys = ["ministral", "gemini", "qwen3-32b", "gpt-oss-120b", "llama-3"]
order_labels = [model_clean[m] for m in modelos_keys]
prompts_ordem = [
    "Zero_Shot",
    "Identity,_Intent_and_Behaviour",
    "Tree_of_Thought",
]
prompts_labels_x = ["Zero-Shot", "IIB", "ToT"]

df_models = df[df["modelo"].isin(modelos_keys)].copy()

# ---------------------------------------------------------------------------
# 2. FIGURA 1: MAPA DE DISPERSÃO
# ---------------------------------------------------------------------------
df_flesch = df[df["metrica"] == "flesch"].copy()
df_wps = df[df["metrica"] == "words_per_sentence"].copy()

merged_scatter = pd.merge(
    df_flesch[["modelo", "prompt", "mean"]],
    df_wps[["modelo", "prompt", "mean"]],
    on=["modelo", "prompt"],
    suffixes=("_flesch", "_wps"),
)

merged_scatter["modelo_lbl"] = merged_scatter["modelo"].map(model_clean)

fig, ax = plt.subplots(figsize=(10.5, 7.2))

x_mid = 17.5
y_mid = 40.0

ax.axvline(x_mid, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
ax.axhline(y_mid, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

ax.fill_between([8, x_mid], y_mid, 67, color="#e6f2ff", alpha=0.35, zorder=0)
ax.text(
    9.1,
    63.8,
    "★ REGIÃO IDEAL\n(Alta Legibilidade & Sintaxe Curta)",
    fontsize=9.5,
    fontweight="bold",
    color="#004085",
    bbox=dict(
        boxstyle="round,pad=0.4", fc="#cce5ff", ec="#b8daff", lw=1, alpha=0.95
    ),
)

for idx, row in merged_scatter.iterrows():
    mod_lbl = row["modelo_lbl"]
    p_key = row["prompt"]

    if pd.isna(mod_lbl):
        if row["modelo"] == "corpus_split":
            mod_lbl = "Original"
        else:
            continue

    m_shape = marker_dict.get(p_key, "o")
    c_color = colors_dict.get(mod_lbl, "#555555")

    ax.scatter(
        row["mean_wps"],
        row["mean_flesch"],
        color=c_color,
        marker=m_shape,
        s=140,
        alpha=0.9,
        edgecolors="k",
        linewidth=0.9,
        zorder=3,
    )

model_legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=k,
        markerfacecolor=v,
        markersize=9,
        markeredgecolor="k",
    )
    for k, v in colors_dict.items()
    if k != "Original"
]

prompt_legend = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="IIB (Quadrado ■)",
        markerfacecolor="gray",
        markersize=9,
        markeredgecolor="k",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        label="ToT (Triângulo ▲)",
        markerfacecolor="gray",
        markersize=9,
        markeredgecolor="k",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Zero-Shot (Círculo ●)",
        markerfacecolor="gray",
        markersize=9,
        markeredgecolor="k",
    ),
]

# Marcador do texto Original (X cinza), incluído na legenda de modelos
model_legend.append(
    Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        label="Original",
        markerfacecolor=colors_dict["Original"],
        markersize=9,
        markeredgecolor="k",
    )
)

leg1 = ax.legend(
    handles=model_legend,
    title="Modelo de Linguagem (Cor)",
    loc="center right",
    bbox_to_anchor=(1.35, 0.72),
    fontsize=8.5,
    title_fontsize=9.5,
)
ax.add_artist(leg1)
leg2 = ax.legend(
    handles=prompt_legend,
    title="Estratégia de Prompt (Forma)",
    loc="center right",
    bbox_to_anchor=(1.38, 0.32),
    fontsize=8.5,
    title_fontsize=9.5,
)

ax.set_title(
    "Mapa de Dispersão Bidimensional (WPS vs. Flesch FRE)",
    fontweight="bold",
    fontsize=12,
    pad=14,
)
ax.set_xlabel(
    "Palavras por Sentença (WPS) [Menor = Frases Mais Curtas ↓]",
    fontweight="bold",
    fontsize=10,
)
ax.set_ylabel(
    "Facilidade de Leitura (Flesch FRE) [Maior = Mais Fácil ↑]",
    fontweight="bold",
    fontsize=10,
)
ax.set_xlim(8, 27)
ax.set_ylim(18, 67)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("grafico_mapa_dispersao_continuo.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3. FIGURA 1b: PAINEL 2x2 DE SIMULAÇÕES AGRUPADAS
# ---------------------------------------------------------------------------
pivot_df = df.pivot_table(
    index=["modelo", "prompt"], columns="metrica", values="mean"
).reset_index()
pivot_df["modelo_lbl"] = pivot_df["modelo"].map(model_clean)
pivot_df["prompt_lbl"] = pivot_df["prompt"].map(prompt_clean)

pivot_calc = pivot_df[pivot_df["modelo"].isin(modelos_keys)].copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

simulations = [
    (
        "words_per_sentence",
        "flesch",
        "1. Flesch FRE vs. WPS (SIMULAÇÃO PRINCIPAL)",
        "WPS (Palavras/Sentença) ↓",
        "Flesch FRE ↑",
        True,
        False,  # invert_y
    ),
    (
        "words_per_sentence",
        "gunning_fox",
        "2. Gunning Fog vs. WPS (REDUNDANTE A — Espelho Inverso)",
        "WPS (Palavras/Sentença) ↓",
        "Gunning Fog ↓ (eixo invertido — topo = melhor)",
        False,
        True,  # invert_y: menor Gunning Fog é melhor, então topo = menor valor
    ),
    (
        "syllables_per_content_word",
        "flesch",
        "3. Flesch FRE vs. Sy/CW (REDUNDANTE B — Componente Embutido)",
        "Sílabas/Palavra (Sy/CW) ↓",
        "Flesch FRE ↑",
        False,
        False,
    ),
    (
        "words_per_sentence",
        "cw_freq_brwac",
        "4. BrWaC Freq vs. WPS (COMPLEMENTAR / VOCABULÁRIO)]",
        "WPS (Palavras/Sentença) ↓",
        "Frequência BrWaC ↑",
        False,
        False,
    ),
]

for ax, (x_col, y_col, title, x_label, y_label, is_main, invert_y) in zip(
    axes.flatten(), simulations
):
    
    r_val, p_val = pearsonr(pivot_calc[x_col], pivot_calc[y_col])
    
    title_with_corr = f"{title}\n[Pearson R = {r_val:+.3f} | p = {p_val:.4f}]"
    
    
    for idx, row in pivot_df.iterrows():
        m_lbl = row["modelo_lbl"]
        p_key = row["prompt"]
        if pd.isna(m_lbl):
            continue

        c = colors_dict.get(m_lbl, "#555555")
        m = marker_dict.get(p_key, "o")

        s = 130 if is_main else 80
        ax.scatter(
            row[x_col],
            row[y_col],
            color=c,
            marker=m,
            s=s,
            alpha=0.88,
            edgecolors="k",
            linewidth=0.8,
        )

    ax.set_title(
        title_with_corr,
        fontweight="bold",
        fontsize=10,
        pad=10,
        color="#003366" if is_main else "#333333",
    )
    ax.set_xlabel(x_label, fontweight="bold", fontsize=9.5)
    ax.set_ylabel(y_label, fontweight="bold", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    if invert_y:
        ax.invert_yaxis()

    if is_main:
        ax.patch.set_facecolor("#f4f8ff")

leg1 = axes[0, 0].legend(
    handles=model_legend[:-1],  # sem "Original" aqui
    title="Modelo (Cor)",
    loc="lower right",
    fontsize=8,
    title_fontsize=8.5,
)
axes[0, 0].add_artist(leg1)
axes[0, 0].legend(
    handles=prompt_legend,
    title="Prompt (Forma)",
    loc="upper right",
    fontsize=8,
    title_fontsize=8.5,
)

fig.suptitle(
    "Comparação das Simulações de Dispersão Agrupadas",
    fontsize=12,
    fontweight="bold",
    y=0.99,
)

plt.tight_layout()
plt.savefig("grafico_simulacoes_agrupadas.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 4. RECONSTRUÇÃO DOS DADOS BRUTOS (PARA OS VIOLIN PLOTS)
# ---------------------------------------------------------------------------
df_filtered_flesch = df[
    (df["metrica"] == "flesch") & (~df["modelo"].isin(sabias))
].copy()

reconstructed_rows = []
for idx, row in df_filtered_flesch.iterrows():
    q_eval = np.linspace(0.05, 0.95, 10)
    xp = [0, 0.25, 0.5, 0.75, 1.0]
    fp = [row["min"], row["q25"], row["median"], row["q75"], row["max"]]
    pts = np.interp(q_eval, xp, fp)

    for val in pts:
        reconstructed_rows.append({
            "modelo": row["modelo"],
            "prompt": row["prompt"],
            "valor": val,
        })

df_raw = pd.DataFrame(reconstructed_rows)
df_raw["modelo_lbl"] = df_raw["modelo"].map(model_clean)

# ---------------------------------------------------------------------------
# 5. FIGURA 2: VIOLIN PLOT GERAL
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

sns.violinplot(
    data=df_raw,
    x="modelo_lbl",
    y="valor",
    inner="quartile",
    palette="Set2",
    order=order_labels,
    ax=ax,
    density_norm="width",
    bw_adjust=0.8,
    cut=1,
)

sns.stripplot(
    data=df_raw,
    x="modelo_lbl",
    y="valor",
    color="black",
    size=6,
    jitter=0.12,
    alpha=0.75,
    order=order_labels,
    ax=ax,
)

violin_explanation = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=7,
        label="Pontos Pretos: Pontuações dos Documentos/Amostras",
    ),
    Line2D(
        [0],
        [0],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Linhas Tracejadas Internas: Mediana & Quartis ($Q_1, Q_3$)",
    ),
]

ax.legend(
    handles=violin_explanation, loc="upper left", fontsize=8, frameon=True
)

ax.set_title(
    "Distribuição Global da Legibilidade (Flesch FRE ↑) - Todos os"
    " Prompts Agrupados",
    fontweight="bold",
    fontsize=11,
    pad=10,
)
ax.set_xlabel("Modelo de Linguagem", fontweight="bold", fontsize=10)
ax.set_ylabel(
    "Facilidade de Leitura (Flesch FRE) ↑\n[Maior = Mais Fácil]",
    fontweight="bold",
    fontsize=9.5,
)
ax.set_xticklabels(order_labels, rotation=15, ha="right", fontsize=9)
ax.set_ylim(-2, 85)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("grafico_violin_geral.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 6. FIGURA 2b: VIOLIN PLOTS - PROMPTS EMPILHADOS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 13))

prompts_panels = [
    ("Zero_Shot", "Estratégia Zero-Shot"),
    (
        "Identity,_Intent_and_Behaviour",
        "Estratégia IIB (Identity, Intent & Behaviour)",
    ),
    ("Tree_of_Thought", "Estratégia ToT (Tree of Thought)"),
]

for idx, (ax, (p_key, p_title)) in enumerate(zip(axes, prompts_panels)):
    p_df = df_raw[df_raw["prompt"] == p_key]

    sns.violinplot(
        data=p_df,
        x="modelo_lbl",
        y="valor",
        inner="quartile",
        palette="Set2",
        order=order_labels,
        ax=ax,
        density_norm="width",
        bw_adjust=0.8,
        cut=1,
    )

    sns.stripplot(
        data=p_df,
        x="modelo_lbl",
        y="valor",
        color="black",
        size=6,
        jitter=0.12,
        alpha=0.75,
        order=order_labels,
        ax=ax,
    )

    if idx == 0:
        ax.legend(
            handles=violin_explanation, loc="upper left", fontsize=8, frameon=True
        )

    ax.set_title(p_title, fontweight="bold", fontsize=11, pad=8)
    ax.set_xlabel("Modelo de Linguagem", fontweight="bold", fontsize=10)
    ax.set_ylabel(
        "Facilidade de Leitura (Flesch FRE) ↑", fontweight="bold", fontsize=9
    )
    ax.set_xticklabels(order_labels, rotation=10, ha="right", fontsize=9)
    ax.set_ylim(-2, 85)
    ax.grid(True, linestyle=":", alpha=0.6)

fig.suptitle(
    "Distribuição da Legibilidade (Flesch FRE ↑) por Estratégia de"
    " Prompt",
    fontsize=12,
    fontweight="bold",
    y=1.01,
)

plt.tight_layout()
plt.savefig("grafico_violin_prompts.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 7. FIGURA 3: RADAR REESTRUTURADO 2x2
# ---------------------------------------------------------------------------
df_all = df[~df["modelo"].isin(sabias)].copy()
metrics_radar_keys = [
    "flesch",
    "gunning_fox",
    "words_per_sentence",
    "cw_freq_brwac",
    "syllables_per_content_word",
]
labels_radar = [
    "Facilidade\n(Flesch FRE ↑)",
    "Acessibilidade\n(1/Gunning Fog ↑)",
    "Frases Curtas\n(1/WPS ↑)",
    "Frequência Lexical\n(BrWaC ↑)",
    "Palavras Curtas\n(1/Sy/CW ↑)",
]

def min_max_scale_global(df_in, metric, invert=False):
    s = df_in[df_in["metrica"] == metric]["mean"]
    min_v, max_v = s.min(), s.max()
    sub = df_in[df_in["metrica"] == metric].copy()
    if invert:
        sub["norm"] = 20 + 80 * (max_v - sub["mean"]) / (max_v - min_v)
    else:
        sub["norm"] = 20 + 80 * (sub["mean"] - min_v) / (max_v - min_v)
    return sub

df_radar_base = df_all[df_all["modelo"].isin(modelos_keys)].copy()
norm_dfs = []
for m, inv in [
    ("flesch", False),
    ("gunning_fox", True),
    ("words_per_sentence", True),
    ("cw_freq_brwac", False),
    ("syllables_per_content_word", True),
]:
    norm_dfs.append(min_max_scale_global(df_radar_base, m, invert=inv))

df_norm_all = pd.concat(norm_dfs)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 11),
    subplot_kw=dict(polar=True, theta_offset=np.pi / 2, theta_direction=-1),
)
axes_flat = axes.flatten()

prompts_radar_2x2 = [
    ("Zero_Shot", "Estratégia: Zero-Shot"),
    (
        "Identity,_Intent_and_Behaviour",
        "Estratégia: IIB (Identity, Intent & Behaviour)",
    ),
    ("Tree_of_Thought", "Estratégia: ToT (Tree of Thought)"),
    ("Agregado", "Média Agregada (Todos os Prompts)"),
]

num_vars = len(labels_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for idx, (ax, (p_key, p_title)) in enumerate(zip(axes_flat, prompts_radar_2x2)):
    if p_key == "Agregado":
        pivot_p = df_norm_all.groupby(["modelo", "metrica"])["norm"].mean().unstack()
    else:
        p_data = df_norm_all[df_norm_all["prompt"] == p_key]
        pivot_p = p_data.pivot(index="modelo", columns="metrica", values="norm")

    for mod in modelos_keys:
        mod_lbl = model_clean[mod]
        color = colors_dict[mod_lbl]
        values = pivot_p.loc[mod, metrics_radar_keys].values.tolist()
        values += values[:1]

        ax.plot(angles, values, color=color, linewidth=2.2, label=mod_lbl)
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, size=8, fontweight="bold")
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=7, color="gray")
    ax.set_ylim(0, 105)
    ax.set_title(p_title, fontweight="bold", pad=16, fontsize=11)

    if idx == 0:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.25, 1.15),
            ncol=1,
            fontsize=8.5,
            title="Modelo de Linguagem (Cores)",
            title_fontsize=9,
            frameon=True,
        )

fig.suptitle(
    "Perfis Multidimensionais de Simplificação (Layout 2x2 com Agregado)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout()
plt.savefig("grafico_radar_paineis_vertical.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close()

print("[SUCESSO] Todos os 5 arquivos PDF foram gerados!")