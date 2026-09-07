import text_metrics
import json
import os

dir = "textos"

if os.path.exists(dir):
    for path in os.listdir(dir):
        if path.endswith(".json"):
            with open(f"{dir}/{path}", f"r", encoding="utf-8") as f_in:
                modelo = path.split(".")[0]
                
                dados_json = json.load(f_in)
                
                pasta_saida = f"saida_{modelo}"
                if not os.path.exists(pasta_saida):
                    os.makedirs(pasta_saida)
                
                for chave_prompt in dados_json:
                    lista_textos = dados_json[chave_prompt]
                    
                    for i, texto_simplificado in enumerate(lista_textos):
                        
                        raw = texto_simplificado.encode("utf-8", "surrogateescape").decode("utf-8")
                        t = text_metrics.Text(raw)
                        
                        ret = text_metrics.no_palavras_metrics.values_for_text(t).as_flat_dict()
                        
                        nome_arquivo = f"metrics_{modelo}_{chave_prompt.replace(' ', '_')}_{i}.json"
                        caminho_final = os.path.join(pasta_saida, nome_arquivo)
                        
                        with open(caminho_final, "w", encoding="utf-8") as f_out:
                            json.dump(ret, f_out, ensure_ascii=False, indent=4)
                            
print("Cálculo de métricas concluído!")