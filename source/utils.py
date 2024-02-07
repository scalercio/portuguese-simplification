import numpy as np
import torch
from source.paths import get_data_filepath, MODELS_DIR, get_dataset_dir
TEST_DATASET = 'porsimplessent'

def get_evaluate_kwargs(language, phase='valid'):
    return {
        ('en', 'valid'): {'test_set': 'asset_valid'},
        ('en', 'test'): {'test_set': 'asset_test'},
        ('fr', 'valid'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('alector', 'valid', 'complex'),
            'refs_sents_paths': [get_data_filepath('alector', 'valid', 'simple')],
        },
        ('fr', 'test'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('alector', 'test', 'complex'),
            'refs_sents_paths': [get_data_filepath('alector', 'test', 'simple')],
        },
        ('es', 'valid'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('simplext_corpus', 'valid', 'complex'),
            'refs_sents_paths': [get_data_filepath('simplext_corpus', 'valid', 'simple')],
        },
        ('es', 'test'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('simplext_corpus', 'test', 'complex'),
            'refs_sents_paths': [get_data_filepath('simplext_corpus', 'test', 'simple')],
        },
        ('pt', 'valid'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath(TEST_DATASET, 'valid', 'complex'),
            'refs_sents_paths': [get_data_filepath(TEST_DATASET, 'valid', 'simple')],
            'src_exemplars': get_data_filepath(TEST_DATASET, 'train', 'complex'),
            'tgt_exemplars': get_data_filepath(TEST_DATASET, 'train', 'simple'),
            'beta': 8,
        },
        ('pt', 'test'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath(TEST_DATASET, 'test', 'complex'),
            'refs_sents_paths': [get_data_filepath(TEST_DATASET, 'test', 'simple')],
            'src_exemplars': get_data_filepath(TEST_DATASET, 'train', 'complex'),
            'tgt_exemplars': get_data_filepath(TEST_DATASET, 'train', 'simple'),
            'beta': 12,
        },
    }[(language, phase)]

"""get output"""


def peek_weights(model):
    for i, k in model.named_parameters():
        if "block.2.layer.0.SelfAttention.k.weight" in i:
            print(i)
            print(k)


def tokenize(input, tokenizer, sent_length):
    return tokenizer(input, max_length=sent_length, truncation=True, padding="max_length", return_tensors="pt").input_ids.cuda()


def peek_output(input, context, model, tokenizer):
    print("input:", input)
    print("context:", context)
    input_ids = tokenize(input)
    context_ids = tokenize(context)
    extractor_output = model.net.get_extractor_output(
        use_cache_context_ids=context_ids)
    # print(extractor_output)
    outputs = model.net.generate(
        input_ids=input_ids, use_cache_extractor_outputs=extractor_output, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def peek_transfer_output(input, target_examplars, origin_examplars, model, tokenizer):
    targets = ()
    for sent in target_examplars:
        targets += (tokenize(sent),)
    origins = ()
    for sent in origin_examplars:
        origins += (tokenize(sent),)
    input_ids = tokenize(input)
    extractor_output = model.net.get_extractor_output(
        input_ids=input_ids, use_cache_origin_examplars_ids=origins, use_cache_target_examplars_ids=targets)
    outputs = model.net.generate(
        input_ids=input_ids, use_cache_extractor_outputs=extractor_output, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


"""module and training"""


def drop_noise(sent, drop_rate=0.2):
    for i in range(int(((sent > 1).sum() * drop_rate))):
        randIdx = np.random.choice(np.where((sent > 1).cpu())[0])
        sent = torch.concat((sent[:randIdx], sent[randIdx + 1:]))
    return sent


def rand_token(tokenizer):
    special_tokens_set = set(tokenizer.all_special_ids)
    t = np.random.randint(tokenizer.vocab_size)
    if t in special_tokens_set:
        return rand_token(tokenizer)
    return t


def add_noise(sent, tokenizer, drop_rate=0.4):
    for i in range(int(((sent > 1).sum() * drop_rate))):
        randIdx = np.random.choice(np.where((sent > 1).cpu())[0])
        sent = torch.concat((sent[:randIdx], torch.tensor(
            [rand_token(tokenizer)]).cuda(), sent[randIdx:]))
    return sent


def pad_sent(sent, sent_length):
    target = sent_length
    if sent.shape[0] > target:
        return sent[:target]
    return torch.concat((sent, torch.zeros(target - sent.shape[0], dtype=torch.long).cuda()))

# def drop_noise_(sent, drop_rate=0.4):
#   for i in range(int(sent.shape[0] * drop_rate)):
#     randIdx = np.random.randint(sent.shape[0])
#     sent = torch.concat((sent[:randIdx], sent[randIdx + 1:]))
#   return sent


def apply_noise(sents, tokenizer, sent_length):
    res = ()
    for i, sent in enumerate(sents):
        sent = drop_noise(sent)
        sent = add_noise(sent, tokenizer)
        sent = pad_sent(sent, sent_length)
        res += (sent,)
    return torch.vstack(res)

def get_outputs_unchanged(simples, sources):
    assert len(simples)==len(sources)
    count = 0
    for (simple, original) in zip(simples, sources):
        if simple.lower().strip() == original.lower().strip():
            print(simple)
            count=count+1
    return count*100/len(simples)
