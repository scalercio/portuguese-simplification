import requests
from bs4 import BeautifulSoup
import json

URL_NITEROISEPLAG = "https://egg.seplag.niteroi.rj.gov.br/dicionario-de-linguagem-simples/"
PROCESSED_DATA_PATH = "linguagem_simples_niteroiseplag_processed_data.json"

response = requests.get(URL_NITEROISEPLAG)
webpage_content = response.content
soup = BeautifulSoup(webpage_content, "html.parser")

# find all the div elements with the class "htb-modal-body"
div_elements = soup.find_all("div", class_="htb-modal-body")

data = []
id = 1

for div in div_elements:
    term = div.find("h3").text.strip()
    p_elements = div.find_all("p")

    definicao_tecnica = p_elements[0].text.strip()
    definicao_simples = p_elements[1].text.strip()
    linguagem_simples = p_elements[2].text.strip()

    entry1 = {
        "id": id,
        "original": f"{term}: {definicao_tecnica.replace('Definição técnica:', '').strip()}",
        "simplified": f"{term}: {definicao_simples.replace('Definição simples:', '').strip()}"
    }
    id+=1

    entry2 = {
        "id": id,
        "original": f"{term}: {definicao_tecnica.replace('Definição técnica:', '').strip()}",
        "simplified": f"{term}: {linguagem_simples.replace('Linguagem simples:', '').strip()}"
    }
    id+=1

    entry3 = {
        "id": id,
        "original": f"{term}: {definicao_simples.replace('Definição simples:', '').strip()}",
        "simplified": f"{term}: {linguagem_simples.replace('Linguagem simples:', '').strip()}"
    }
    id+=1

    data.extend([entry1, entry2, entry3])

json_data = json.dumps(data, ensure_ascii=False, indent=4)

with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as file:
    file.write(json_data)
