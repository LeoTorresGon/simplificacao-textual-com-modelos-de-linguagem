import json

text_list = []
MIN_PALAVRAS = 300
MAX_PALAVRAS = 500

with open("corpus.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

for text in texts:
    palavras = text.split()
    
    if len(palavras) <= MAX_PALAVRAS:
        text_list.append(text)
        continue
        
    while len(palavras) > 0:
        if len(palavras) <= MAX_PALAVRAS:
            text_list.append(" ".join(palavras))
            break
            
        janela_busca = palavras[MIN_PALAVRAS:MAX_PALAVRAS]
        
        ponto_encontrado = False
        for i in range(len(janela_busca) - 1, -1, -1):
            if "." in janela_busca[i]:
                indice_corte = MIN_PALAVRAS + i + 1
                ponto_encontrado = True
                break
                
        if not ponto_encontrado:
            indice_corte = MAX_PALAVRAS
            
        trecho = " ".join(palavras[:indice_corte])
        text_list.append(trecho)
        
        palavras = palavras[indice_corte:]

with open(f"corpus_split-{MIN_PALAVRAS}_{MAX_PALAVRAS}.json", "w", encoding="utf-8") as f:
    json.dump(text_list, f, ensure_ascii=False, indent=4)

