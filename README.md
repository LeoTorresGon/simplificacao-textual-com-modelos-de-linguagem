# simplificacao-textual-com-modelos-de-linguagem

Esse repositório guarda os principais códigos utilizados durante a pesquisa de viabilidade de modelos de linguagem para simplificação textual com fins de inclusão e acessibilidade, uma pesquisa de iniciação científica feita por mim e orientada pelo Prof. Michel Carlo Rodrigues Leles.

## Guia de arquivos

- SAW.py
Arquivo referente ao cálculo do método SAW (Simple Additive Weighting). A entrada dele é o arquivo com todos os dados organizados em um csv com colunas modelo,prompt,metrica,count,mean,std,median,iqr,q25,q75,min,max.

- api.py
Arquivo usado para consumo da API do GroqCloud.

- graficos_final.py
Arquivo que gera os gráficos exibidos no artigo. A entrada é a mesma do SAW.py.

- ministral.py
Arquivo responsável pela execução do modelo Ministral 3 no Supercomputador SantosDummont.

- nilc.py
Arquivo responsável pela coleta das métricas do pacote NILC.

- sabia7B.py
Arquivo usado para executar o modelo Sabia 1 no Supercomputador SantosDummont

- split_palavras.py
Script que separa os textos iniciais em blocos com tamanho definido.

- statistics.py
Script que gera as estatísticas a partir dos dados brutos gerados pelos modelos.
