import csv
import json

RAW_DATA_PATH = "linguagem_simples_tjrs_raw_data.txt"
PROCESSED_DATA_PATH = "linguagem_simples_tjrs_processed_data.json"

def txt_to_json(csv_filename, json_filename):
    data = []
    with open(csv_filename, mode='r', encoding='utf-8') as csvfile:
        sentences = csvfile.readlines()
        
        for i, row in enumerate(sentences):
            #print(row)
            if row.strip():
                row = row.strip().split('\t')
                tech_definition = row[0]
                simple_definition = row[1]

                original = f"{tech_definition}"
                simplified = f"{simple_definition}"

                data.append({
                    "id": i + 1,
                    "original": original,
                    "simplified": simplified
                })
    
    with open(json_filename, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    txt_to_json(RAW_DATA_PATH, PROCESSED_DATA_PATH)
