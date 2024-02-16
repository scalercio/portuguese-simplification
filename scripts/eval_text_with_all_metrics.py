# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
import wandb
from lightning.pytorch.loggers import WandbLogger
#from pytorch_lightning.loggers import TensorBoardLogger
import copy
import os

from source.utils import get_evaluate_kwargs, get_outputs_unchanged
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from bert_score import score

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

    testset = True
    dataset = 'museu'
    config = {
        'evaluate_kwargs': get_evaluate_kwargs("pt",'test') if testset else get_evaluate_kwargs("pt"),
        'testset': testset,
        'simplifications_path': 'museu-test-simplification-pt/angelic-violet-17/0_0'#'data/asset/muss-test-simplification'#'test-simplification-pt/luminous-snake-18/0_0'#'test-simplification-pt/twinkling-lamp-19/0_0'#'test-simplification-pt/winter-frost-15/0_0',#, 
    }
    if 'asset' not in dataset:        
        with open(config['evaluate_kwargs']['refs_sents_paths'][0], 'r') as f1, open(config['simplifications_path'], 'r') as f2, open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f3:
            ref_seq = f1.readlines()    
            simple_seq = f2.readlines()
            src_seq = f3.readlines()
    else:
        with open(config['simplifications_path'], 'r') as f2, open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f3:
            simple_seq = f2.readlines()
            src_seq = f3.readlines()
        ref_seq = []
        for i in range(10):
            with open(f'data/asset/test/simp{i}.txt') as f:
                ref_seq.append(f.readlines())
        

    print(len(simple_seq))
    if 'asset' not in dataset:
        assert len(simple_seq) == len(ref_seq)
    else:
        assert len(simple_seq) == len(ref_seq[9])
    
    assert len(simple_seq) == len(src_seq)
    results={}
    results['sari'] = corpus_sari(orig_sents=src_seq,
                       sys_sents=simple_seq,
                       refs_sents=[ref_seq])
    results['bleu'] = corpus_bleu(sys_sents=simple_seq,
                       refs_sents=[ref_seq],
                       lowercase=True)
    if 'asset' in dataset:
        P, R, F1 = score(simple_seq, list(map(list, zip(*ref_seq))), lang = 'pt', verbose = True)
    else:
        P, R, F1 = score(simple_seq, ref_seq, lang = 'pt', verbose = True)
    results['bert_score'] = F1.mean()
    results['outputs_unchanged'] = get_outputs_unchanged(simple_seq,src_seq)
    print(results)
