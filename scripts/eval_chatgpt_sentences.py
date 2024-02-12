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
from source.utils import get_evaluate_kwargs
from easse.sari import corpus_sari
import random
from source.helpers import write_lines

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    """# 1. Prepare Data"""
    print(torch.__version__)
    print(torch.version.cuda)
    # set random seed
    rand_seed = 123
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

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
    for tipo_one_shot in types:
        for i in range(3):
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

            simplified_sentences=[]
            for sentence_dict in data:
                #print(sentence_dict['simplified'])
                simplified_sentences.append(sentence_dict['simplified'])

            assert len(src_seq) == len(simplified_sentences)
            all_sentences.append(simplified_sentences)
            sari = corpus_sari(orig_sents=src_seq,
                               sys_sents=simplified_sentences,
                               refs_sents=[ref_seq])
            result +=sari
            print(sari)
        print(result/3)
        print('Fim desse tipo')
        result=0       
    #logger.log({'sari':0})
    human_selection = []
    random.seed(777)
    print(len(all_sentences[0]))
    for i in range(len(all_sentences[0])):
        text = all_sentences[random.choice(range(len(all_sentences)))][i]
        print(f'{i}_{text}')
        human_selection.append(text)
        #print(all_sentences[0][i],all_sentences[1][i])
    write_lines(human_selection, f'/home/arthur/nlp/repo/simplification/portuguese-simplification/data/porsimplessent/chatgpt/human_evaluation.txt')
    wandb.finish()
