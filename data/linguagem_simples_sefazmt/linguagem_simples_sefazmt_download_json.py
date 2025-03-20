import requests
from bs4 import BeautifulSoup
import json

URL_SEFAZMT = "https://www5.sefaz.mt.gov.br/dicionario-clone"
PROCESSED_DATA_PATH = "linguagem_simples_sefazmt_processed_data.json"

response = requests.get(URL_SEFAZMT)
webpage_content = response.content
soup = BeautifulSoup(webpage_content, "html.parser")

div_elements_content = soup.find_all("div", style="font-weight: 400;")
div_elements_term = soup.find_all(lambda tag: tag.name == 'div' and 'mosaic-color-white' in tag.get('class', []))

data = []
id = 1

def preprocess_text(text):
    """ Limpa e prepara o texto removendo caracteres especiais e espaços duplos. """
    text = text.replace('\t', ' ')  # Troca caracteres de tabulação por espaços
    text = text.replace('\n', ' ')  # Troca quebras de linha por espaços
    text = text.replace('\u00a0', ' ').replace('\u200b', ' ').replace('\u200c', ' ').replace('\u200d', ' ').replace('–', ' ')
    text = ' '.join(text.split())   # Remove espaços duplos, triplos, etc
    return text

for term, content in zip(div_elements_term, div_elements_content):
    term_text = term.find("h2").text.strip()
    paragraphs = content.find_all("p")

    for paragraph in paragraphs:
        paragraph_text = paragraph.get_text()
        if "Definição Técnica:" in paragraph_text:
            definicao_tecnica = paragraph_text.replace("Definição Técnica:", "").strip()
        elif "Linguagem simples:" in paragraph_text:
            linguagem_simples = paragraph_text.replace("Linguagem simples:", "").strip()

    data.append({
        "id": id,
        "original": preprocess_text(f"{term_text}: {definicao_tecnica}"),
        "simplified": preprocess_text(f"{term_text}: {linguagem_simples}")
    })

    id += 1

json_data = json.dumps(data, ensure_ascii=False, indent=4)

with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as file:
    file.write(json_data)
