import datetime
import json
from transformers import LlamaTokenizer, LlamaForCausalLM

local_model_path = "/scratch/ufsj/hpc4agents-br/leonardo.silva/simptext/sabia-7b-local"

tokenizer = LlamaTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = LlamaForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=True
)

def query_llm(payload):
    inputs = tokenizer(payload['inputs'], return_tensors="pt").to(model.device)
    
    output = model.generate(
        inputs["input_ids"],
        max_length=2048,
        eos_token_id=tokenizer.eos_token_id 
    )
    
    output_tokens = output[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(output_tokens, skip_special_tokens=True)


output = []
with open("/scratch/ufsj/hpc4agents-br/leonardo.silva/simptext/texts.json", "r") as f:
    texts = json.load(f)
    for input_text in texts:
        prompt = (
        f"""Atue como um especialista em Linguagem Cidadã e Acessibilidade Textual. 
        Sua tarefa é reescrever o texto técnico abaixo seguindo as diretrizes de "Linguagem Simples" para garantir a inclusão de cidadãos com diferentes níveis de letramento.

        Diretrizes de Saída:
        1. Substitua termos jurídicos ou técnicos (jargões) por palavras do cotidiano.
        2. Utilize a ordem direta (Sujeito + Verbo + Complemento).
        3. Use frases curtas e apenas uma ideia por parágrafo.
        4. Mantenha o sentido original e a precisão das informações.
        5. Evite siglas sem explicação e construções na voz passiva.

        Exemplo de Entrada: A presente lei é redigida de forma técnica e clara. Na lei revogada, a primeira leitura ao art. 5º dava a errônea noção de que toda locação se findava quando terminado o prazo do contrato. No entanto, quando a antiga lei cuidava das locações residenciais, estatuía no seu art. 52 as hipóteses em que o despejo era permitido. Assim, ficava possível o pedido de despejo, independentemente de qualquer motivação (a chamada denúncia vazia), nas locações não residenciais em três situações: naquelas por prazo indeterminado, mediante prévia notificação com prazo de trinta dias; naquelas por prazo determinado, findo este, nesse caso independentemente de notificação, e naquelas por tempo determinado, que houvessem se prorrogado por prazo indeterminado. Nessa última situação, entendeu o extinto 2º Tribunal de Alçada Civil de São Paulo: “É dispensável a notificação premonitória, quando o pedido de retomada de prédio não residencial se dá logo após o término do contrato, notadamente se a ação é ajuizada dentro em 30 (trinta) dias” (Súmula nº 14). As locações residenciais, findo o prazo determinado, continuavam a ter vigência por prazo indeterminado e só admitiam o pedido de despejo motivado dentro das hipóteses legais. As situações de despejo promovidas pelo novo adquirente do imóvel, ou quando da extinção de usufruto ou fideicomisso, embora tratadas também como hipóteses de denúncia vazia, deviam ser consideradas à margem do sistema geral (o mesmo sucede na presente Lei, como examinamos, arts. 7º e 8º).1  Esta Lei abre possibilidades mais amplas de denúncia vazia ou imotivada. Esse aspecto é, na verdade, o ponto crucial do atual ordenamento, possibilitando uma dinâmica maior, mais real e menos demagógica no mercado imobiliário e na questão habitacional. As novas locações subordinam-se a esse novo sistema (as locações residenciais celebradas anteriormente à lei atual têm disciplina própria nas disposições transitórias, arts. 77 e 78)
        
        Exemplo de Saída: A lei antiga gerava dúvidas porque parecia que o aluguel acabava logo que o prazo do contrato vencia. Na prática, a regra funcionava de formas diferentes para casas e comércios. Nos aluguéis de casas, o contrato continuava valendo mesmo após o fim do prazo e o dono só podia pedir o imóvel de volta se tivesse um motivo justo previsto em lei. Já nos aluguéis comerciais, o dono podia retomar o imóvel sem apresentar uma justificativa, o que é chamado de denúncia vazia. Isso era permitido em três situações: quando o contrato não tinha data de término e o dono avisava o inquilino com 30 dias de antecedência; no dia exato em que o contrato acabava; ou logo após o fim do prazo, caso o inquilino continuasse no local. A justiça decidiu que o dono nem precisava enviar um aviso se pedisse o imóvel de volta nos primeiros 30 dias após o fim do contrato. Outros casos, como a venda do imóvel para um novo proprietário, também permitem pedir a saída do inquilino sem uma justificativa específica. A lei atual ampliou essas possibilidades de retomar o imóvel sem motivo, o que torna o mercado imobiliário mais ágil e realista. Todos os novos aluguéis seguem esse sistema moderno, enquanto os contratos de casas feitos antes da lei atual obedecem a regras de transição específicas.

        Entrada: {input_text}

        Saída em Linguagem Cidadã:"""
        )
        output.append(query_llm({"inputs": prompt}))

with open(f"/scratch/ufsj/hpc4agents-br/leonardo.silva/simptext/outputs_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)