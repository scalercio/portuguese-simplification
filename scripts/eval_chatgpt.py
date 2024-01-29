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
    prompt = 'kewetal'
    repeat = False
    model_version = "chatgpt"
    dataset = 'ccnet'
    config = {
        'prompt': prompt,
        'repeat': repeat,
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
    assert len(src_seq) == len(ref_seq)
    result = 0
    for i in range(3):
        if repeat:
            simplified_file = f'data/porsimplessent/chatgpt/simplified_chatgpt_{prompt}_zeroshot_repete_{i+1}.json'
        else:
            simplified_file = f'data/porsimplessent/chatgpt/simplified_chatgpt_{prompt}_zeroshot_{i+1}.json'
            
        print(simplified_file)
        with open(simplified_file) as f3:
            data=json.load(f3)
            
        simplified_sentences=[]
        for sentence_dict in data:
            simplified_sentences.append(sentence_dict['simplified'])
        
        assert len(src_seq) == len(simplified_sentences)
        sari = corpus_sari(orig_sents=src_seq,
                           sys_sents=simplified_sentences,
                           refs_sents=[ref_seq])
        result +=sari
        print(sari)
    print(result/3)       
    #logger.log({'sari':0})
    wandb.finish()
