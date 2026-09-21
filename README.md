# Simplificação Textual com Modelos de Linguagem

Repositório dos códigos, dados e análises desenvolvidos durante uma pesquisa de iniciação científica sobre o uso de **modelos de linguagem generativa (LLMs)** para **simplificação textual em português**, com foco em **Linguagem Cidadã, acessibilidade e inclusão**.

O projeto investiga diferentes modelos de linguagem e estratégias de prompting para transformar textos técnicos e complexos em textos mais acessíveis, preservando suas informações e seu significado original.

A pesquisa está vinculada ao projeto **"Inteligência Artificial Generativa Aplicada à Linguagem Cidadã: Simplificação de Textos Públicos para Inclusão e Acessibilidade"**, desenvolvido na Universidade Federal de São João del-Rei (UFSJ).

---

## Sobre o projeto

Textos jurídicos, administrativos e institucionais frequentemente utilizam vocabulário técnico, estruturas sintáticas complexas e construções pouco familiares ao público geral. Essa característica pode dificultar o acesso da população a informações relacionadas a direitos, serviços públicos e decisões institucionais.

A pesquisa investiga a utilização de **Inteligência Artificial Generativa** como ferramenta para auxiliar nesse processo, avaliando diferentes modelos de linguagem e diferentes estratégias de instrução (*prompting*) para a reescrita dos textos.

Entre as tarefas consideradas pelo projeto estão:

* Simplificação lexical e sintática;
* Reescrita em Linguagem Cidadã;
* Avaliação da legibilidade dos textos;
* Análise quantitativa das características linguísticas das respostas dos modelos.

Essas tarefas fazem parte da metodologia definida para avaliar modelos de linguagem open-source e multilíngues aplicados à acessibilidade textual.

---

## Objetivos

### Objetivo geral

Investigar o uso de técnicas de Inteligência Artificial Generativa para promover a acessibilidade da linguagem jurídica, educacional e institucional em língua portuguesa por meio da simplificação textual.

### Objetivos específicos

* Construir e organizar um corpus de textos complexos;
* Realizar experimentos com diferentes modelos de linguagem;
* Comparar diferentes estratégias de *prompting*;
* Aplicar métricas automáticas de legibilidade e características linguísticas;
* Analisar estatisticamente os resultados;
* Utilizar métodos de decisão multicritério para comparar as alternativas;
* Produzir análises e visualizações dos resultados;
* Disponibilizar os códigos e dados utilizados na pesquisa para favorecer a reprodutibilidade.

---

## Metodologia

O experimento é composto por diferentes etapas.

### 1. Preparação do corpus

Os textos utilizados nos experimentos são organizados e divididos em trechos para processamento pelos modelos.

O script `split_palavras.py` realiza essa divisão, procurando manter os trechos dentro de uma faixa de tamanho definida e priorizando cortes em finais de frases.

A metodologia do projeto prevê a utilização de um corpus composto principalmente por documentos jurídicos, administrativos e educacionais, incluindo também conteúdos como fluxogramas e diagramas institucionais.

### 2. Geração das simplificações

Os textos são processados por diferentes modelos de linguagem utilizando estratégias distintas de *prompting*.

Entre as estratégias utilizadas estão:

* **Zero-Shot**
* **Identity, Intent and Behaviour (IIB)**
* **Tree of Thought (ToT)**

Os prompts utilizados estão disponíveis em:

```text
dados/prompts.json
```

As instruções procuram orientar os modelos para características de Linguagem Simples, como:

* utilização de vocabulário cotidiano;
* redução de jargões;
* utilização de ordem direta;
* frases mais curtas;
* utilização de voz ativa;
* explicação de siglas;
* preservação das informações do texto original.

### 3. Modelos avaliados

O repositório contém experimentos envolvendo diferentes modelos de linguagem, incluindo:
Ministral 3 (3B), Llama 3.1 (8B), Qwen3-32B, GPT-OSS 120B, Gemini Flash 3.5, Sabiá 1 e Sabiá 4.

Os experimentos foram realizados utilizando tanto modelos executados localmente em infraestrutura computacional quanto modelos acessados por API.

O projeto de pesquisa originalmente previa experimentos comparativos entre modelos open-source em português e modelos multilíngues com suporte ao português.

---

## Execução dos modelos

### Modelos locais

Os scripts abaixo foram utilizados para execução de modelos em ambiente computacional de alto desempenho:

```text
ministral.py
sabia7B.py
```

Esses scripts utilizam modelos armazenados localmente e bibliotecas do ecossistema Hugging Face/Transformers.

### Modelos via API

O arquivo:

```text
api.py
```

realiza chamadas à API da **Groq** para geração das simplificações.

O modelo configurado no script é:

```text
openai/gpt-oss-120b
```

Esse valor deve ser alterado para utilizar os outros modelos disponíveis pela API.

A chave da API deve ser fornecida por meio da variável de ambiente:

```env
GROQ_API_KEY=sua_chave_aqui
```

**Não coloque sua chave diretamente no código ou no repositório.**

---

## Avaliação dos textos

Após a geração dos textos simplificados, as respostas dos modelos são avaliadas por meio de métricas linguísticas.

O arquivo:

```text
nilc.py
```

utiliza o pacote **NILC-Metrix** para calcular as métricas dos textos gerados.

Entre as métricas utilizadas na análise estão:

* Flesch;
* Words per Sentence;
* Sentences per Paragraph;
* Syllables per Content Word;
* Content Word Frequency;
* Noun Ratio;
* Type-Token Ratio (TTR);
* Gunning's Fog Index;
* Brunet's Index;
* Honoré's Statistic.

Essas métricas permitem analisar diferentes aspectos dos textos, incluindo legibilidade, comprimento das frases, complexidade lexical e características linguísticas.

A proposta de pesquisa também estabelece métricas como **Flesch, FKGL, BLEU e SARI** como possibilidades de avaliação automática.

---

## Análise estatística

O arquivo:

```text
statistics.py
```

processa os resultados produzidos pelas métricas e gera estatísticas consolidadas para os diferentes modelos e estratégias de *prompting*.

Os dados incluem medidas como:

* média;
* desvio padrão;
* mediana;
* intervalo interquartil;
* primeiro quartil;
* terceiro quartil;
* mínimo;
* máximo.

Os resultados consolidados são armazenados em arquivos JSON e CSV dentro do diretório `dados/`.

O relatório estatístico utilizado nas análises possui a seguinte estrutura:

```text
modelo
prompt
metrica
count
mean
std
median
iqr
q25
q75
min
max
```

---

## Análise multicritério

Além da análise individual das métricas, o projeto utiliza métodos de **decisão multicritério** para combinar diferentes características de legibilidade em uma análise conjunta.

### SAW

O arquivo:

```text
SAW.py
```

implementa o método **Simple Additive Weighting (SAW)**.

São utilizados cinco critérios:

| Critério                   | Peso | Tipo      |
| -------------------------- | ---: | --------- |
| Flesch                     | 0,35 | Benefício |
| Words per Sentence         | 0,25 | Custo     |
| Gunning-Fog                | 0,20 | Custo     |
| Content Word Frequency     | 0,15 | Benefício |
| Syllables per Content Word | 0,05 | Custo     |

Os critérios são normalizados e combinados por meio de uma soma ponderada.

O resultado é salvo em:

```text
dados/ranking_multicriterio_saw.csv
```

### TOPSIS

O arquivo:

```text
multicriterio_topsis.py
```

implementa uma análise utilizando o método **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**.

O método utiliza os mesmos cinco critérios principais e os respectivos pesos para calcular a proximidade de cada alternativa em relação às soluções ideal positiva e negativa.

---

## Visualização dos resultados

O arquivo:

```text
graficos_final.py
```

é responsável pela geração dos gráficos utilizados na análise dos resultados.

Entre as visualizações estão análises relacionando diferentes métricas de legibilidade e comparando modelos e estratégias de *prompting*.

Os gráficos são construídos a partir do arquivo:

```text
dados/relatorio_estatistico_ic.csv
```

---

## Estrutura do repositório

```text
simplificacao-textual-com-modelos-de-linguagem/
│
├── README.md
│
├── api.py
├── SAW.py
├── graficos_final.py
├── ministral.py
├── multicriterio_topsis.py
├── nilc.py
├── sabia7B.py
├── split_palavras.py
└── statistics.py
│
└── dados/
    ├── analise_consolidada_modelos.json
    ├── analise_consolidada_modelos_filtrada.json
    ├── corpus_split.json
    ├── dados_brutos_para_violinplot.csv
    ├── estatisticas_completas_por_modelo_prompt.json
    ├── estatisticas_por_prompt_geral.json
    ├── estatisticas_por_prompt_sem_sabia.json
    ├── medias_consolidadas_por_modelo.json
    ├── medias_consolidadas_por_modelo_filtrada.json
    ├── medias_consolidadas_por_prompt.json
    ├── medias_consolidadas_por_prompt_filtrada.json
    ├── medias_consolidadas_por_prompt_sem_sabia.json
    ├── prompts.json
    ├── ranking_multicriterio_saw.csv
    └── relatorio_estatistico_ic.csv
```

---

## Fluxo de processamento

De forma simplificada, o fluxo utilizado no projeto pode ser representado como:

```text
                ┌──────────────────┐
                │ Corpus de textos │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Divisão dos      │
                │ textos           │
                └────────┬─────────┘
                         │
                         ▼
              ┌───────────────────────┐
              │ Modelos de linguagem  │
              │ + estratégias de      │
              │ prompting             │
              └───────────┬───────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Textos           │
                │ simplificados    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ NILC-Metrix      │
                │ métricas         │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Análise          │
                │ estatística      │
                └────────┬─────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      ┌──────────────┐       ┌──────────────┐
      │ Visualização │       │ SAW / TOPSIS │
      └──────────────┘       └──────────────┘
```

---

## Dados

O diretório `dados/` contém os resultados intermediários e consolidados dos experimentos.

### Dados brutos e estatísticos

```text
relatorio_estatistico_ic.csv
dados_brutos_para_violinplot.csv
```

contêm dados utilizados para análises estatísticas e visualizações.

### Médias consolidadas

Os arquivos:

```text
medias_consolidadas_por_modelo.json
medias_consolidadas_por_prompt.json
```

organizam as métricas agregadas por modelo e por estratégia de *prompting*.

Também estão disponíveis versões filtradas e versões que excluem os experimentos relacionados ao Sabiá.

### Prompts

Os prompts utilizados nos experimentos estão disponíveis em:

```text
dados/prompts.json
```

Isso permite consultar e reproduzir as diferentes estratégias de instrução utilizadas na pesquisa.

---

## Reprodutibilidade

O projeto busca manter os scripts, dados e análises organizados de forma a permitir a reprodução dos experimentos.

A proposta original da pesquisa estabelece a disponibilização dos scripts, dados e fluxos em repositório aberto, seguindo os princípios **FAIR — Findable, Accessible, Interoperable e Reusable**.

Alguns scripts presentes neste repositório possuem caminhos específicos de infraestrutura, especialmente os scripts destinados à execução dos modelos no ambiente HPC. Portanto, esses caminhos precisam ser adaptados para a infraestrutura utilizada.

---

## Tecnologias utilizadas

O projeto utiliza principalmente:

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **SciPy**
* **NILC-Metrix**
* **Groq API**

---

## Pesquisa

**Projeto:** Inteligência Artificial Generativa Aplicada à Linguagem Cidadã: Simplificação de Textos Públicos para Inclusão e Acessibilidade

**Instituição:** Universidade Federal de São João del-Rei (UFSJ)

**Programa:** PIBIC

**Área:** Ciência da Computação

**Palavras-chave:** Inteligência Artificial Generativa, Linguagem Cidadã, Acessibilidade Textual e Inclusão Digital.

