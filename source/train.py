from pathlib import Path;import sys;sys.path.append(str(Path(__file__).resolve().parent.parent)) # fix path
import time
from transformers import T5TokenizerFast
import pytorch_lightning as pl
from pytorch_lightning import Trainer
#import wandb
from lightning.pytorch.loggers import WandbLogger

from source.data import CCNetDataModule
from source.model import (T5ForConditionalGenerationWithExtractor,
                   TextSettrModel)
from source.preprocessor import Preprocessor
import numpy as np
import torch
torch.set_float32_matmul_precision('medium')


def run_training(config, features_kwargs, dataset='ccnet'):
    print(torch.__version__)
    print(torch.version.cuda)
    # set random seed
    rand_seed = 123
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    preprocessor = Preprocessor(features_kwargs)
    preprocessor.preprocess_dataset(dataset)
    #config["dataset"] = dataset
    
    tokenizer = T5TokenizerFast.from_pretrained(config['model_version'])
    module = CCNetDataModule(config['batch_size'], tokenizer, config['sent_length'])
    logger = WandbLogger(
        project="simplification-pt-v2",
        job_type=f'ptt5-base_{config["sent_length"]:03d}',
        config=config            
    )
    
    if config['load_ckpt'] is None:
        model = TextSettrModel(**config, tokenizer=tokenizer)
    else:
        model = TextSettrModel.load_from_checkpoint(config['load_ckpt'],
                                                    **config, tokenizer=tokenizer)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=root, filename='{epoch}')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="sari", save_top_k = 2, mode = 'max')
    #logger = TensorBoardLogger("logs", name="textual_simplification")
    trainer = Trainer(max_epochs = 15, default_root_dir='./', val_check_interval=0.1, precision='bf16', logger=logger,
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