# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import wandb
import json
from lightning.pytorch.loggers import WandbLogger
from easse.sari import corpus_sari
import warnings
warnings.filterwarnings("ignore")
from source.paths import get_data_filepath

def calculate_best_sentence_counts(best_indices, num_files, types):
    counts = [0] * num_files
    for idx in best_indices:
        counts[idx] += 1
    
    print(f"{dataset}: Number of sentences added in best_sentences_{dataset}.txt per source file:")
    for type_idx, tipo in enumerate(types):
        for i in range(3):
            file_index = type_idx * 3 + i
            print(f"{tipo} {i+1}: {counts[file_index]} sentences")

def final_sari_for_each_dataset_with_best_sentences(ref_seq, src_seq, best_sentences):
    final_sari = corpus_sari(
        orig_sents=src_seq,
        sys_sents=best_sentences,
        refs_sents=[ref_seq]
    )
    print(f"Final SARI score for {dataset}: {final_sari}")

if __name__ == '__main__':
    prompt = 'fengetal'
    repeat = True
    few_shot = False
    one_shot = True
    types = ['sintática', 'anáfora', 'ordem', 'redundante_lexical']
    model_version = "chatgpt"
    datasets = ['museu', 'porsimplessent']
    
    for dataset in datasets:
        config = {
        'prompt': prompt,
        'repeat': repeat,
        'few_shot': few_shot,
        'one_shot': one_shot,
        'evaluate_kwargs': {'refs_sents_paths': [get_data_filepath(dataset, 'test', 'simple')], 'orig_sents_path': get_data_filepath(dataset, 'test', 'complex'),},
        'model_version': model_version,
        }

        logger = WandbLogger(
        project="chatgpt-simplification-pt",
        job_type=f'chatgpt_prompt:{prompt}_repeat:{repeat}',
        reinit = True,
        mode = 'disabled',
        config=config
        )

        with open(config['evaluate_kwargs']['refs_sents_paths'][0], 'r') as f1, open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f2:
            ref_seq = f1.readlines()
            src_seq = f2.readlines()

        best_sentences = []
        best_indices = []
        num_files = len(types) * 3

        for idx, src_sentence in enumerate(src_seq):
            ref_sentences = [ref_seq[idx]]
            simplified_sentences = []

            for tipo_one_shot in types:
                for i in range(3):
                    if 'museu' in dataset:
                        simplified_file = f'data/{dataset}/chatgpt/one_shot_feng/simplified_gpt-3.5-turbo-instruct_fengetal_one_shot_repete_{tipo_one_shot}_complex.test.{dataset}_{i+1}.json'
                    else:
                        simplified_file = f'data/{dataset}/chatgpt/one_shot_feng/simplified_gpt-3.5-turbo-instruct_fengetal_one_shot_repete_{tipo_one_shot}_{i+1}.json'

                    with open(simplified_file) as f3:
                        data = json.load(f3)

                    simplified_sentences.append(data[idx]['simplified'])

            best_sari = -1
            best_sentence = ""
            best_index = -1

            for i, sys_sentence in enumerate(simplified_sentences):
                sari_score = corpus_sari(
                    orig_sents=[src_sentence],
                    sys_sents=[sys_sentence],
                    refs_sents=[ref_sentences]
                )

                if sari_score > best_sari:
                    best_sari = sari_score
                    best_sentence = sys_sentence
                    best_index = i

            best_sentences.append(best_sentence)
            best_indices.append(best_index)

        output_file = f'data/{dataset}/chatgpt/one_shot_feng/best_sentences_{dataset}.txt'
        with open(output_file, 'w') as f_out:
            for sentence in best_sentences:
                f_out.write(f"{sentence}\n")
        print(f"Best sentences for {dataset} saved in: {output_file}")

        calculate_best_sentence_counts(best_indices, num_files, types)

        final_sari_for_each_dataset_with_best_sentences(ref_seq, src_seq, best_sentences)

        wandb.finish()
