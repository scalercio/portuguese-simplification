# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import numpy as np
import torch
import wandb
import json
from lightning.pytorch.loggers import WandbLogger
from source.utils import get_evaluate_kwargs, get_outputs_unchanged
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from bert_score import score
import random
from source.helpers import write_lines

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # hyperparameters
    prompt = 'fengetal'
    repeat = True
    few_shot = False
    one_shot = True
    types = ['sintática','anáfora', 'ordem', 'redundante_lexical']
    model_version = "chatgpt"
    dataset = 'ccnet'
    config = {
        'prompt': prompt,
        'repeat': repeat,
        'few_shot': few_shot,
        'one_shot': one_shot,
        'evaluate_kwargs': get_evaluate_kwargs("pt", 'test'),
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
        src_seq = f2.readlines()
        ref_seq = f1.readlines()
    #    src_seq = f2.read().split('\n')
    #    ref_seq = f1.read().split('\n')
    #
    #if src_seq[-1] == "":
    #    src_seq=src_seq[:-1]
    #
    #if ref_seq[-1] == "":
    #    ref_seq=ref_seq[:-1]
    #for sent in ref_seq:
    #    print(sent)
    print(len(src_seq))
    assert len(src_seq) == len(ref_seq)
    assert not one_shot == few_shot
    result = 0
    all_sentences = []
    ref_final = []
    src_final = []
    for tipo_one_shot in types:
        for i in range(3):
            ref_final.extend(ref_seq)
            src_final.extend(src_seq)
            if 'feng' in prompt and few_shot:
                simplified_file = f"data/porsimplessent/chatgpt/few_shot_feng/simplified_gpt-3.5-turbo-instruct_fengetal_few_shot_repete_4few_{i+1}.json"
            elif few_shot:
                simplified_file = f"data/porsimplessent/chatgpt/few_shot_kew/simplified_gpt-3.5-turbo-instruct_kewetal_few_shot_repete_4few_{i+1}.json"
            elif 'feng' in prompt:
                simplified_file = f'data/porsimplessent/chatgpt/one_shot_feng/simplified_gpt-3.5-turbo-instruct_fengetal_one_shot_repete_{tipo_one_shot}_{i+1}.json'
            else:
                simplified_file = f'data/porsimplessent/chatgpt/one_shot_kew/simplified_gpt-3.5-turbo-instruct_kewetal_one_shot_repete_{tipo_one_shot}_{i+1}.json'

            print(simplified_file)
            with open(simplified_file) as f3:
                data=json.load(f3)

            for sentence_dict in data:
                #print(sentence_dict['simplified'])
                all_sentences.append(sentence_dict['simplified'])

    assert len(ref_final) == len(all_sentences)
    assert len(ref_final) == len(src_final)
    print(len(ref_final)/12)

    results={}
    
    results['sari'] = corpus_sari(orig_sents=src_final,
                       sys_sents=all_sentences,
                       refs_sents=[ref_final])
    
    results['bleu'] = corpus_bleu(sys_sents=all_sentences,
                       refs_sents=[ref_final],
                       lowercase=True)
    P, R, F1 = score(all_sentences, ref_final, lang = 'pt', verbose = True)
    results['bert_score'] = F1.mean()
    results['outputs_unchanged'] = get_outputs_unchanged(all_sentences,src_final)
    print(results)

    wandb.finish()
