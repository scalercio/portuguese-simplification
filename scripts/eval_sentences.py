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
    dataset = 'museu'#'complex.test.asset.txt_'
    config = {
        'prompt': prompt,
        'repeat': repeat,
        'few_shot': few_shot,
        'one_shot': one_shot,
        'evaluate_kwargs': get_evaluate_kwargs("pt", 'test'),
        'model_version': model_version,
    }
    if 'asset' not in dataset:        
        with open(config['evaluate_kwargs']['refs_sents_paths'][0], 'r') as f1, open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f2:
            src_seq = f2.readlines()
            ref_seq = f1.readlines()
        
        #with open('museu-test-simplification-pt/festive-firecracker-7/0_0', 'r') as f3:
        with open('data/museu/muss-test-simplification', 'r') as f3:
            all_sentences = f3.readlines()
    else:
        with open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f2:
            src_seq = f2.readlines()
            
        ref_seq = []
        for i in range(10):
            with open(f'data/asset/test/simp{i}.txt') as f:
                ref_seq.append(f.readlines())
                
    print(len(src_seq))
    if 'asset' in dataset:
        assert len(src_seq) == len(ref_seq[9])
    else:
        assert len(src_seq) == len(ref_seq)


    results={}
    #for i in range(len(src_seq)):
    #    
    #    results['sari'] = corpus_sari(orig_sents=[src_seq[i]],
    #                       sys_sents=[all_sentences[i]],
    #                       refs_sents=[[ref_seq[i]]])
    #
    #    results['bleu'] = corpus_bleu(sys_sents=[all_sentences[i]],
    #                       refs_sents=[[ref_seq[i]]],
    #                       lowercase=True)
    #    if 'asset' in dataset:#to do quando for o asset
    #        P, R, F1 = score(all_sentences[i], list(map(list, zip(*ref_seq))), lang = 'pt', verbose = True)
    #    else:
    #        P, R, F1 = score([all_sentences[i]], [ref_seq[i]], lang = 'pt', verbose = True)
    #    results['bert_score'] = F1.mean()
    #    results['outputs_unchanged'] = get_outputs_unchanged(all_sentences,src_seq)
    #    with open('data/museu/sari_values_muss', 'a') as f1, open('data/museu/bleu_values_muss', 'a') as f2, open('data/museu/bscore_values+muss', 'a') as f3:
    #        print('{:.8f}'.format(results['sari']), file=f1)
    #        print('{:.8f}'.format(results['bleu']), file=f2)
    #        print('{:.8f}'.format(results['bert_score']), file=f3)
    #        
    #        
    #    print(i)
    results['sari'] = corpus_sari(orig_sents=src_seq,
                               sys_sents=all_sentences,
                               refs_sents=[ref_seq])

    results['bleu'] = corpus_bleu(sys_sents=all_sentences,
                       refs_sents=[ref_seq],
                       lowercase=True)
    if 'asset' in dataset:
        P, R, F1 = score(all_sentences[i], list(map(list, zip(*ref_seq))), lang = 'pt', verbose = True)
    else:
        P, R, F1 = score(all_sentences, ref_seq, lang = 'pt', verbose = True)
    results['bert_score'] = F1.mean()
    results['outputs_unchanged'] = get_outputs_unchanged(all_sentences,src_seq)
    print(results)