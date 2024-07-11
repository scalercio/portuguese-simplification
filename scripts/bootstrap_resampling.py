# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import numpy as np
import torch
import json
torch.set_float32_matmul_precision('medium')

from source.utils import get_evaluate_kwargs, get_outputs_unchanged
from easse.sari import corpus_sari

import warnings
warnings.filterwarnings("ignore")

def generate_sari_confidence_intervals(original_sentences, simplified_sentences, reference_sentences, baseline_seq, iterations=1000):
    """
    Generate multiple SARI score confidence intervals for different test set using bootstrap resampling.
    
    Parameters:
    - original_sentences: List of original sentences.
    - simplified_sentences: List of simplified sentences.
    - reference_sentences: List of reference sentences.    
    - baseline_seq: List of baseline simplification sentences.
    - iterations: Number of bootstrap samples to generate per test sets.
    
    Returns:
    - The statistical significance
    """
    
    sari_scores = []
    for __ in range(iterations):
        indices = np.random.choice(len(original_sentences), len(original_sentences), replace=True)
        sampled_originals = [original_sentences[i] for i in indices]
        sampled_simplifications = [simplified_sentences[i] for i in indices]
        baseline_simplifications = [baseline_seq[i] for i in indices]

        sampled_references = [[reference_sentences[i] for i in indices]]
        #sampled_references = [[reference_sentences[i]] for i in indices]

        score = corpus_sari(orig_sents=sampled_originals, sys_sents=sampled_simplifications, refs_sents=sampled_references)
        #print(score)
        score_baseline = corpus_sari(orig_sents=sampled_originals, sys_sents=baseline_simplifications, refs_sents=sampled_references)
        #print(score_baseline)
        if score > score_baseline:
            sari_scores.append(1)
        else:
            sari_scores.append(0)

        #sari_scores.sort()
        ## calculate the indices for the 2.5th and 97.5th %
        #lower_index = max(0, int(0.025 * len(sari_scores)))
        #upper_index = min(len(sari_scores) - 1, int(0.975 * len(sari_scores)))

        #all_intervals.append((sari_scores[lower_index], sari_scores[upper_index]))

    return sum(sari_scores)/iterations

def compare_with_muss(dataset, **config):
    if 'asset' not in dataset:        
        with open(config['evaluate_kwargs']['refs_sents_paths'][0], 'r') as f1, open(config['simplifications_path'], 'r') as f2, open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f3, open(config['simplifications_baseline_path'], 'r') as f4:
            ref_seq = f1.readlines()
            simple_seq = f2.readlines()
            src_seq = f3.readlines()
            baseline_seq = f4.readlines()
    else:
        raise ValueError("Bootstrap resampling not implemented for Asset dataset")
        

    print(len(simple_seq))
    assert len(simple_seq) == len(ref_seq)
    assert len(simple_seq) == len(src_seq)
    assert len(simple_seq) == len(baseline_seq)

    confidence_intervals = generate_sari_confidence_intervals(src_seq, simple_seq, ref_seq, baseline_seq)
    print(f"Generated SARI score confidence intervals: {confidence_intervals}")
    
def compare_with_gpt(dataset, **config):
    if 'asset' in dataset:
        raise ValueError("Bootstrap resampling not implemented for Asset dataset")
    
    with open(config['evaluate_kwargs']['refs_sents_paths'][0], 'r') as f1, open(config['simplifications_path'], 'r') as f2, open(config['evaluate_kwargs']['orig_sents_path'], 'r') as f3:
            ref_seq = f1.readlines()
            simple_seq = f2.readlines()
            src_seq = f3.readlines()

    if 'museu' in dataset:
        dataset_string = f'complex.test.{dataset}_'
    elif 'asset' in dataset:
        dataset_string = 'complex.test.asset.txt_'
    else:
        dataset_string = ''
        
    types = ['sintática','anáfora', 'ordem', 'redundante_lexical']
    gpt_sentences = []
    ref_final = []
    src_final = []
    simple_final = []
    for tipo_one_shot in types:
        for i in range(3):
            ref_final.extend(ref_seq)
            src_final.extend(src_seq)
            simple_final.extend(simple_seq)
            simplified_file = f'data/{dataset}/chatgpt/one_shot_feng/simplified_gpt-3.5-turbo-instruct_fengetal_one_shot_repete_{tipo_one_shot}_{dataset_string}{i+1}.json'
            
            print(simplified_file)
            with open(simplified_file) as f3:
                data=json.load(f3)

            for sentence_dict in data:
                #print(sentence_dict['simplified'])
                gpt_sentences.append(sentence_dict['simplified'])
    
    print(len(simple_final))
    assert len(simple_final) == len(ref_final)
    assert len(simple_final) == len(src_final)
    assert len(simple_final) == len(gpt_sentences)
    
    confidence_intervals = generate_sari_confidence_intervals(src_final, simple_final, ref_final, gpt_sentences)
    print(f"Generated SARI score confidence intervals: {confidence_intervals}")       

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
    baseline = 'gpt'
    config = {
        'evaluate_kwargs': get_evaluate_kwargs("pt",'test') if testset else get_evaluate_kwargs("pt"),
        'simplifications_baseline_path': f'data/{dataset}/muss-test-simplification' if 'muss' in baseline else None,
        'simplifications_path': 'test-simplification-pt/twinkling-lamp-19/0_0' if 'porsimplessent' in dataset else 'museu-test-simplification-pt/festive-firecracker-7/0_0' 
    }
    if 'muss' in baseline:
        compare_with_muss(dataset, **config)
    else:
        compare_with_gpt(dataset, **config)
    
