import os
import json

def find_json_files(directory):
    """Encontra todos os arquivos JSON em uma pasta e suas subpastas."""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("data.json"):
                json_files.append(os.path.join(root, file))
    return json_files

def process_json_files(json_files):
    """Extrai as chaves 'original' e 'simplified' de cada arquivo JSON."""
    original_sentences = []
    simplified_sentences = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for entry in data:
                original_sentences.append(entry["original"])
                simplified_sentences.append(entry["simplified"])
    
    return original_sentences, simplified_sentences

def write_to_file(filepath, lines):
    """Escreve as linhas em um arquivo."""
    with open(filepath, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')

def main(directory):
    """Função principal que coordena a operação."""
    # Encontra todos os arquivos JSON
    json_files = find_json_files(directory)
    
    # Processa os arquivos JSON e extrai os textos
    original_sentences, simplified_sentences = process_json_files(json_files)
    
    # Escreve o conteúdo em arquivos
    write_to_file('test.complex', original_sentences)
    write_to_file('test.simple', simplified_sentences)

if __name__ == "__main__":
    # Diretório de entrada (substitua pelo caminho da pasta que contém os arquivos JSON)
    directory = '/home/arthur/nlp/repo/simplification/portuguese-simplification/data'
    main(directory)
