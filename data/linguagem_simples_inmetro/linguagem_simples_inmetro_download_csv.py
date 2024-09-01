import requests
from bs4 import BeautifulSoup
import csv
import re

URL_INMETRO = "https://www.gov.br/inmetro/pt-br/centrais-de-conteudo/publicacoes/dicionario-linguagem-simples/de-a-a-z/index"
RAW_DATA_PATH = "linguagem_simples_inmetro_raw_data.csv"

def fetch_and_parse_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup


def extract_terms(soup):
    terms = []
    headlines = soup.find_all('h2', class_='headline')
    for headline in headlines:
        term = headline.get_text(strip=True)
        terms.append(term)
    return terms


def clean(element):
    for a_tag in element.find_all('a'):
        a_tag.decompose()
    text = element.get_text(strip=True)
    text = text.replace('\u00a0', ' ').replace('\u200b', ' ').replace('\u200c', ' ').replace('\u200d', ' ').replace('–', ' ') # replace NO-BREAK SPACE and other non-breaking spaces w/ a regular space
    text = text.replace(' ()', '').replace('()', '')
    text = re.sub(r'\s+', ' ', text) # remove double spaces
    return text


def extract_definitions(soup):
    definitions = []
    tables = soup.find_all('table', class_='plain')
    
    for table in tables:
        rows = table.find_all('tr')

        tech_definition_img = rows[0].find('img', alt="Emoji preocupado (Termo técnico)")
        if tech_definition_img:
            tech_definition_element = rows[0].find('p')
            tech_definition = clean(tech_definition_element)

            simple_definition_img = rows[1].find('img', alt="Emoji Feliz (Linguagem Simples)")
            if simple_definition_img:
                simple_definition_p = rows[1].find_all('p')
                simple_definition = clean(simple_definition_p[1])
                
                definitions.append([tech_definition, simple_definition])
    
    return definitions


def save_to_csv(terms, definitions, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Termo", "Definição Técnica", "Linguagem Simples"])
        for term, definition in zip(terms, definitions):
            writer.writerow([term] + definition)


if __name__ == "__main__":
    soup = fetch_and_parse_url(URL_INMETRO)
    definitions = extract_definitions(soup)
    terms = extract_terms(soup)
    save_to_csv(terms, definitions, RAW_DATA_PATH)
