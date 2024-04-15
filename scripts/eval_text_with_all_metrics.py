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

def generate_sari_confidence_intervals(original_sentences, simplified_sentences, reference_sentences, iterations=1000, total_test_sets=100):
    """
    Generate multiple SARI score confidence intervals for different test set using bootstrap resampling.
    
    Parameters:
    - original_sentences: List of original sentences.
    - simplified_sentences: List of simplified sentences.
    - reference_sentences: List of reference sentences.
    - iterations: Number of bootstrap samples to generate per test sets.
    - total_test_sets: Number of different test set to evaluate.
    
    Returns:
    - List of confidence intervals for each test sets.
    """
    all_intervals = []
    
    for _ in range(total_test_sets):
        sari_scores = []
        for __ in range(iterations):
            indices = np.random.choice(len(original_sentences), len(original_sentences), replace=True)
            sampled_originals = [original_sentences[i] for i in indices]
            sampled_simplifications = [simplified_sentences[i] for i in indices]

            # verify if reference_sentences is a flat list or a list of list
            if isinstance(reference_sentences[0], list):
                sampled_references = [reference_sentences[i] for i in indices] # Please, review @arthur
            else:
                sampled_references = [[reference_sentences[i]] for i in indices]

            score = corpus_sari(orig_sents=sampled_originals, sys_sents=sampled_simplifications, refs_sents=sampled_references)
            sari_scores.append(score)

        sari_scores.sort()
        # calculate the indices for the 2.5th and 97.5th %
        lower_index = max(0, int(0.025 * len(sari_scores)))
        upper_index = min(len(sari_scores) - 1, int(0.975 * len(sari_scores)))

        all_intervals.append((sari_scores[lower_index], sari_scores[upper_index]))

    return all_intervals

def plot_sari_confidence_intervals(all_intervals):
    """
    Plot the SARI score confidence intervals.
    
    Parameters:
    - all_intervals: List of tuples representing the confidence intervals.
    """
    #TODO
    pass

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

    sari_bootstrap_resampling_intervals = generate_sari_confidence_intervals(src_seq, simple_seq, ref_seq)
    print(f"Generated SARI score confidence intervals: {sari_bootstrap_resampling_intervals[:10]}...")  # Print *only* first 10 intervals
    plot_sari_confidence_intervals(sari_bootstrap_resampling_intervals)

    results={}
    results['bleu'] = corpus_bleu(sys_sents=simple_seq,
                       refs_sents=[ref_seq],
                       lowercase=True)
    if 'asset' in dataset:
        results['sari'] = corpus_sari(orig_sents=src_seq, sys_sents=simple_seq, refs_sents=ref_seq) # Please, review @arthur
        P, R, F1 = score(simple_seq, list(map(list, zip(*ref_seq))), lang = 'pt', verbose = True)
    else:
        # refs_sents should be a "list of list of reference sentences (shape = (n_references, n_samples))
        # https://github.com/feralvam/easse/blob/6a4352ec299ed03fda8ee45445ca43d9c7673e89/easse/sari.py#L242C5-L242C88
        results['sari'] = corpus_sari(orig_sents=src_seq, sys_sents=simple_seq, refs_sents=[[ref] for ref in ref_seq])
        P, R, F1 = score(simple_seq, ref_seq, lang = 'pt', verbose = True)
    results['bert_score'] = F1.mean()
    results['outputs_unchanged'] = get_outputs_unchanged(simple_seq,src_seq)
    print(results)
