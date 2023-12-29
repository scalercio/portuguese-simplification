#import hfai_env
#import hfai
#from hfai.pl import ModelCheckpointHF
#from hfai.pl import HFAIEnvironment
import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
from transformers import T5TokenizerFast
import pytorch_lightning as pl
from pytorch_lightning import Trainer
#import wandb
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

#hfai_env.set_env('barlowtwins')

if __name__ == '__main__':
    """# 1. Prepare Data"""
    print(torch.__version__)
    print(torch.version.cuda)
    # set random seed
    rand_seed = 123
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    # hyperparameters
    batch_size = 8
    sent_length = 50    

    #model = T5ForConditionalGenerationWithExtractor.from_pretrained(
    #    "./pretrained_model/t5-base-with-extractor")
    #model.extractor = copy.deepcopy(model.encoder)
    #model.extractor.is_extractor = True
    #model.lambda_factor = lambda_factor

    # training loop
    lambda_val = 0
    delta_val = 1e-4
    rec_val = 0
    lr = 3e-4
    model_version = "unicamp-dl/ptt5-large-portuguese-vocab"
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
    tokenizer = T5TokenizerFast.from_pretrained(model_version)
    module = CCNetDataModule(batch_size, tokenizer, sent_length)
    logger = WandbLogger(
        project="simplification-pt-v2",
        job_type=f'ptt5-base_{sent_length:03d}',
        config=config            
    )
    
    if config['load_ckpt'] is None:
        model = TextSettrModel(**config, tokenizer=tokenizer)
    else:
        model = TextSettrModel.load_from_checkpoint(config['load_ckpt'],
                                                    **config, tokenizer=tokenizer)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=root, filename='{epoch}')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="sari", save_top_k = 5, mode = 'max')
    #logger = TensorBoardLogger("logs", name="textual_simplification")
    trainer = Trainer(max_epochs = 20, default_root_dir='./', val_check_interval=0.1, precision='bf16', logger=logger,
                          devices = 1, callbacks=[checkpoint_callback], num_sanity_val_steps=0, gradient_clip_val=5)
        #trainer = Trainer(max_epochs=10, gpus=8, default_root_dir="", val_check_interval=0.25,
        #                  precision=32, logger=logger, plugins=[HFAIEnvironment()], callbacks=[cb])

        #model = hfai.pl.nn_to_hfai(model)  # 替换成幻方算子

        #ckpt_path = f'{output_dir}/barlow-twins-lambda-{lambda_val}-{cb.CHECKPOINT_NAME_SUSPEND}.ckpt'
        #ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(
        model,
        module
    )
