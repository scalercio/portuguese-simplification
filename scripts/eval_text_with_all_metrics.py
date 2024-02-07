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
    dataset = 'ccnet'
    config = {
        'evaluate_kwargs': get_evaluate_kwargs("pt",'test') if testset else get_evaluate_kwargs("pt"),
        'testset': testset,
        'simplifications_path': 'test-simplification-pt/restful-sea-4/0_0',
    }
    with open(config['evaluate_kwargs']['refs_sents_paths'][0], 'r') as f1, open(config['simplifications_path'], 'r') as f2, open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f3:
        ref_seq = f1.readlines()    
        simple_seq = f2.readlines()
        src_seq = f3.readlines()
        
    print(len(simple_seq))
    assert len(simple_seq) == len(ref_seq)
    assert len(simple_seq) == len(src_seq)
    results={}
    results['sari'] = corpus_sari(orig_sents=src_seq,
                       sys_sents=simple_seq,
                       refs_sents=[ref_seq])
    results['bleu'] = corpus_bleu(sys_sents=simple_seq,
                       refs_sents=[ref_seq],
                       lowercase=True)
    P, R, F1 = score(simple_seq, ref_seq, lang = 'pt', verbose = True)
    results['bert_score'] = F1.mean()
    results['outputs_unchanged'] = get_outputs_unchanged(simple_seq,src_seq)
    print(results)
