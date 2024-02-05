# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import numpy as np
import torch
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
    complex_sentences=[]
    simple_sentences=[]
    for i in range(42):
        print(i)
        with open(f"data/museu/doc_{i}.original", 'r') as f1, open(f"data/museu/doc_{i}.simple", 'r') as f2:
            complex_seqs = f1.readlines()
            simple_seqs = f2.readlines()
        assert len(complex_seqs) == len(simple_seqs)
        for complex_seq, simple_seq in zip(complex_seqs,simple_seqs):
            if complex_seq.strip()=='':
                continue
            if complex_seq[0]=='C':
                #print(complex_seq.strip().split('   '))
                #print(simple_seq.strip())
                assert complex_seq.strip().split('   ')[1] == simple_seq.strip()
                continue
            if complex_seq[0]=='D':
                assert simple_seq.strip()==''
                continue
            complex_sentences.append(complex_seq.strip().split('   ')[1])
            simple_sentences.append(simple_seq.strip())
    assert len(simple_sentences) == len(complex_sentences)
    write_lines(complex_sentences, f'/home/arthur/nlp/repo/simplification/portuguese-simplification/data/museu/complex')
    write_lines(simple_sentences, f'/home/arthur/nlp/repo/simplification/portuguese-simplification/data/museu/simple')
                

        
        
       
  
    #human_selection = []
    #random.seed(777)
    #print(len(all_sentences[0]))
    #for i in range(len(all_sentences[0])):
    #    text = all_sentences[random.choice(range(len(all_sentences)))][i]
    #    print(f'{i}_{text}')
    #    human_selection.append(text)
    #    #print(all_sentences[0][i],all_sentences[1][i])
    #write_lines(human_selection, f'/home/arthur/nlp/repo/simplification/portuguese-simplification/data/porsimplessent/chatgpt/human_evaluation.txt')
