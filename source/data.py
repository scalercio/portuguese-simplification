from torch.utils.data import Dataset, DataLoader
import ast
import pytorch_lightning as pl
import time

class TrainCCNetDataset(Dataset):
    def __init__(self, src_inst, tgt_inst, tgt_context_inst, tokenizer, sent_length):
        self.src_inst = src_inst
        self.tgt_inst = tgt_inst
        self.tgt_context_inst = tgt_context_inst
        self.tokenizer = tokenizer
        self.sent_length = sent_length

    def __len__(self):
        return len(self.src_inst)

    def to_token(self, sentence):
        tokenized = self.tokenizer(
            [sentence],
            truncation=True,
            max_length=self.sent_length,
            padding='max_length',
            return_tensors="pt"
        )
        ids = tokenized["input_ids"].squeeze()
        mask = tokenized["attention_mask"].squeeze()
        return (ids, mask)

    def __getitem__(self, index):
        return self.to_token(self.tgt_context_inst[index]) + self.to_token(self.tgt_inst[index]) + self.to_token(self.src_inst[index])

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
        tokenized = self.tokenizer(
            [sentence],
            truncation=True,
            max_length=self.sent_length,
            padding='max_length',
            return_tensors="pt"
        )
        ids = tokenized["input_ids"].squeeze()
        mask = tokenized["attention_mask"].squeeze()
        return (ids, mask)
    #def to_token(self, sentence):
    #    return self.tokenizer.encode(sentence, max_length=self.sent_length,
    #                                 truncation=True, padding="max_length",
    #                                 return_tensors="pt")[0]

    def __getitem__(self, index):
        return self.to_token(self.tgt_context_inst[index]) + self.to_token(self.tgt_inst[index])


class CCNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tokenizer, sent_length):
        super().__init__()
        self.tokenizer = tokenizer

        train_src, train_tgt, train_tgt_context = self.read_insts('train')
        valid_src, valid_tgt, valid_tgt_context = self.read_insts('valid')
        infer_src, infer_tgt, _ = self.read_insts('inference')
        print('[Info] {} instances from train set'.format(len(train_src)))
        print('[Info] {} instances from valid set'.format(len(valid_src)))
        print('[Info] {} instances from inference set'.format(len(infer_src)))
        
        self.train = TrainCCNetDataset(
            train_src, train_tgt, train_tgt_context, tokenizer, sent_length)
        self.val = CCNetDataset(
            infer_src, infer_tgt, infer_src, tokenizer, sent_length)
        self.test = CCNetDataset(valid_src, valid_tgt, valid_tgt_context, tokenizer, sent_length)
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

        if mode not in 'inference':            
            src_dir = 'data/ccnet/{}.{}'.format(mode, 'simple')
            #src_dir = 'resources/processed_data/b6e484f0eec4c8c7bccb24a5d0cbe432/ccnet/{}.{}'.format(mode, 'simple')
            tgt_dir = 'data/ccnet/{}.{}'.format(mode, 'complex')
            #tgt_dir = 'resources/processed_data/b6e484f0eec4c8c7bccb24a5d0cbe432/ccnet/{}.{}'.format(mode, 'complex')
            tgt_context_dir = 'data/ccnet/{}.{}'.format(mode, 'complex_context')
        else:
            src_dir = 'data/porsimplessent/{}.{}'.format('valid', 'complex')
            tgt_dir = 'data/porsimplessent/{}.{}'.format('valid', 'simple')
            #tgt_dir = 'data/porsimplessent/{}.{}'.format('valid', 'complex_predicted_bucketized')
            tgt_context_dir = 'data/porsimplessent/{}.{}'.format('valid', 'complex_context')
            

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
