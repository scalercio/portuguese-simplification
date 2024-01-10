# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from source.train import run_training
from source.utils import get_evaluate_kwargs
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # hyperparameters
    batch_size = 80
    sent_length = 85
    lambda_val = 0
    delta_val = 1e-4
    rec_val = 0
    lr = 1e-4
    model_version = "unicamp-dl/ptt5-base-portuguese-vocab"
    dataset = 'ccnet'
    config = {
        'sent_length': sent_length,
        'batch_size': batch_size,
        'delta_val': delta_val, # ver onde usa
        'lambda_val': lambda_val,
        'rec_val': rec_val,
        'lr': lr,
        'evaluate_kwargs': get_evaluate_kwargs("pt"),
        'model_version': model_version,
        'load_ckpt': None#'simplification-pt/4tnl668c/checkpoints/epoch=9-step=90187.ckpt',
    }
    features_kwargs = {
    'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.8},
    'LevenshteinRatioFeature': {'target_ratio': 0.8},
    'WordRankRatioFeature': {'target_ratio': 0.8},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
}
    run_training(config, features_kwargs, dataset)
