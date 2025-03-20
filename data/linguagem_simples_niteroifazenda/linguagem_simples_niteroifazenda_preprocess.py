import pandas as pd
import json

RAW_DATA_PATH = "linguagem_simples_niteroifazenda_raw_data.csv"
PROCESSED_DATA_PATH = "linguagem_simples_niteroifazenda_processed_data.json"

def preprocess_text(text):
    """ Limpa e prepara o texto removendo caracteres especiais e espaços duplos. """
    text = text.replace('\t', ' ')  # Troca caracteres de tabulação por espaços
    text = text.replace('\n', ' ')  # Troca quebras de linha por espaços
    text = text.replace('\u00a0', ' ').replace('\u200b', ' ').replace('\u200c', ' ').replace('\u200d', ' ').replace('–', ' ')
    text = ' '.join(text.split())   # Remove espaços duplos, triplos, etc
    return text

def csv_to_json(input_csv, output_json):
    """ 
        Gera o json com as combinacoes: DT-DS, DT-LS, DS-LS.
        o Termo é concatenado em todas sentencas.

        DT: Definição técnica
        DS: Definição simples
        LS: Linguagem simples
    """
    df = pd.read_csv(input_csv)

    # Preprocessa cada coluna de texto
    for col in df.columns:
        df[col] = df[col].apply(preprocess_text)

    json_data = []
    id = 1

    # Criando os objetos JSON
    for _, row in df.iterrows():
        termo = row['Termo']
        definicao_tecnica = row['Definição técnica']
        definicao_simples = row['Definição simples']
        linguagem_simples = row['Linguagem simples']

        obj1 = {
            "id": f"{id}",
            "original": f"{termo}: {definicao_tecnica}",
            "simplified": f"{termo}: {definicao_simples}"
        }
        id += 1
        obj2 = {
            "id": f"{id}",
            "original": f"{termo}: {definicao_tecnica}",
            "simplified": f"{termo}: {linguagem_simples}"
        }
        id += 1
        obj3 = {
            "id": f"{id}",
            "original": f"{termo}: {definicao_simples}",
            "simplified": f"{termo}: {linguagem_simples}"
        }
        id += 1

        json_data.extend([obj1, obj2, obj3])

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    csv_to_json(RAW_DATA_PATH, PROCESSED_DATA_PATH)
