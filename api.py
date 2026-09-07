import json
import os
from time import sleep
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

model = "openai/gpt-oss-120b"

with open ("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
    
with open("corpus_split2.json", "r", encoding="utf-8") as f:
    texts = json.load(f)
    
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

saida = {}

counter = 0

print("Iniciando simplificação...")
for prompt_name, prompt in prompts.items():
    outputs = []
    for text in texts:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model=model,
        )
        counter += 1
        print(f"Texto simplificado! ({counter}/{len(texts)})")
        if counter % 5 == 0:
            sleep(30)
        outputs.append(chat_completion.choices[0].message.content)
    saida[prompt_name] = outputs

with open(f"output_{model}.json".replace("/", "_"), "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=4)