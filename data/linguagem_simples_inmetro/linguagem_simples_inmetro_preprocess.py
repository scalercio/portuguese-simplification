import csv
import json

RAW_DATA_PATH = "linguagem_simples_inmetro_raw_data.csv"
PROCESSED_DATA_PATH = "linguagem_simples_inmetro_processed_data.json"

def csv_to_json(csv_filename, json_filename):
    data = []
    with open(csv_filename, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for i, row in enumerate(csvreader):
            term = row['Termo']
            tech_definition = row['Definição Técnica']
            simple_definition = row['Linguagem Simples']
            
            original = f"{term}: {tech_definition}"
            simplified = f"{term}: {simple_definition}"
            
            data.append({
                "id": i + 1,
                "original": original,
                "simplified": simplified
            })
    
    with open(json_filename, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    csv_to_json(RAW_DATA_PATH, PROCESSED_DATA_PATH)
