from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import pytorch_lightning as pl
import time


def get_raw_data():
    raw_data = pd.read_fwf("./data/sampled_999986.csv")
    raw_data = raw_data[["sents"]]
    # raw_data = raw_data.sample(n=5000, random_state = rand_seed)

    print(raw_data["sents"].apply(lambda x: len(x) - 1).sum())
    raw_data["cumlen"] = raw_data["sents"].apply(
        lambda x: len(x) - 1).cumsum() - 1
    raw_data["len"] = raw_data["sents"].apply(lambda x: len(x) - 1)
    raw_data = raw_data.set_index("cumlen")

    return raw_data


class CCNetDataset(Dataset):
    def __init__(self, src_inst, tgt_inst, tgt_context_inst, tokenizer, sent_length):
        self.src_inst = src_inst
        self.tgt_inst = tgt_inst
        self.tgt_context_inst = tgt_context_inst
        self.tokenizer = tokenizer
        self.sent_length = sent_length

    def __len__(self):
        return len(self.src_inst)

    def to_token(self, sentence):
        return self.tokenizer.encode(sentence, max_length=self.sent_length,
                                     truncation=True, padding="max_length",
                                     return_tensors="pt")[0]

    def __getitem__(self, index):
        return self.to_token(self.tgt_context_inst[index]), self.to_token(self.tgt_inst[index]), self.to_token(self.src_inst[index])


class CCNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, sent_length):
        super().__init__()
        self.tokenizer = tokenizer

        train_src, train_tgt, train_tgt_context = self.read_insts('train')
        valid_src, valid_tgt, valid_tgt_context = self.read_insts('valid')
        print('[Info] {} instances from train set'.format(len(self.train_src)))
        print('[Info] {} instances from valid set'.format(len(self.valid_src)))
        
        self.train = CCNetDataset(
            train_src, train_tgt, train_tgt_context, tokenizer, sent_length)
        self.test = CCNetDataset(
            valid_src, valid_tgt, valid_tgt_context, tokenizer, sent_length)
        self.val = CCNetDataset(valid_src, valid_tgt, valid_tgt_context, tokenizer, sent_length)
        self.batch_size = batch_size

    def read_insts(self, mode):
        """
        Read instances from input file
        Args:
            mode (str): 'train' or 'valid'.
            shuffle (bool): whether randomly shuffle training data.
            opt: it contains the information of transfer direction.
        Returns:
            src_seq: list of the lists of token ids for each source sentence.
            tgt_seq: list of the lists of token ids for each tgrget sentence.
        """

        src_dir = 'data/ccnet/{}.{}'.format(mode, 'simple')
        tgt_dir = 'data/ccnet/{}.{}'.format(mode, 'complex')
        tgt_context_dir = 'data/ccnet/{}.{}'.format(mode, 'complex_context')

        src_seq, tgt_seq, tgt_context_seq
        with open(src_dir, 'r') as f1, open(tgt_dir, 'r') as f2 , open(tgt_context_dir, 'r') as f3:
            start = time.time()
            src_seq = f1.readlines()
            tgt_seq = f2.readlines()
            tgt_context_seq = f3.readlines()
            end = time.time()
            print("Execution time in seconds: ",(end-start))

            return src_seq, tgt_seq, tgt_context_seq
    
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4)
