import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
from transformers import T5TokenizerFast
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import wandb
from lightning.pytorch.loggers import WandbLogger
#from pytorch_lightning.loggers import TensorBoardLogger
import copy
import os

from data import CCNetDataModule
from model import (T5ForConditionalGenerationWithExtractor,
                   TextSettrModel)
from utils import get_evaluate_kwargs

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
    batch_size = 48
    sent_length = 50    

    tokenizer = T5TokenizerFast.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

    module = CCNetDataModule(batch_size, tokenizer, sent_length)

    # training loop
    paraphrase = False
    lambda_val = 1e-2
    delta_val = 1e-4
    rec_val = 0
    lr = 1e-4
    config = {
        'sent_length': sent_length,
        'batch_size': batch_size,
        'delta_val': delta_val, # ver onde usa
        'lambda_val': lambda_val,
        'rec_val': rec_val,
        'lr': lr,
        'evaluate_kwargs': get_evaluate_kwargs("pt"),
    }
    
    for beta_val in [ 4, 8, 12, 16]:
        config['evaluate_kwargs']['beta'] = beta_val
        logger = WandbLogger(
            project="eval-simplification-pt",
            job_type=f'ptt5-base_{sent_length:03d}',
            reinit = True,
            config=config            
        )
        trainer = Trainer(max_epochs = 15, default_root_dir='./', val_check_interval=0.1, precision='bf16', logger=logger,
                              devices = 1, num_sanity_val_steps=0, gradient_clip_val=5)
        model = TextSettrModel.load_from_checkpoint('simplification-pt/1jcqftal/checkpoints/epoch=14-step=147042.ckpt',
                                                    **config, tokenizer=tokenizer)
        trainer.validate(model, module)
        wandb.finish()
