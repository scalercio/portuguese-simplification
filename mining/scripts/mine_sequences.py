# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import faiss
from tqdm import tqdm

from muss.utils.submitit import get_executor
from muss.utils.helpers import get_file_hash, get_files_hash, log_action, yield_lines
from muss.resources.paths import get_dataset_dir
from muss.laser import get_laser_embeddings
from muss.mining.preprocessing import (
    get_subshard_paths,
    get_sentences_paths,
    sentence_tokenize_subshard,
    split_ccnet_shard,
    create_base_index,
    get_index_name,
)
from muss.mining.nn_search import (
    get_cache_dir,
    get_results_path,
    compute_and_save_nn_batched,
    get_paraphrase_pairs,
    get_pairs_path,
    compute_and_save_simplification_pairs,
    get_index_path,
    compute_and_save_embeddings,
    get_filter_string_representation,
    combine_simplifications_in_dataset,
    get_simplification_pairs_paths,
)
from muss.mining.filtering import SimplicityScorer

ccnet_dir = Path(
    input(
        'Please download the CCNet corpus from https://github.com/facebookresearch/cc_net and enter the path to the downloaded data: '
    )
)
language = input('What language do you want to process? (en/fr/es/pt): ')
cluster = 'local'
dataset_dir = get_dataset_dir('uts') / language
# For large jobs only
slurm_partition = 'debug'
slurm_array_parallelism = 1024

# Split CCNet shards into subshards
with log_action('Splitting CCNet shards into smaller subshards'):
    # We need to split each shard even more for the LASER embeddings to fit in memory
    n_shards = {  # Number of shards to take for each languages for ~1B sentences
        'en': 15,
        'fr': 25,
        'pt': 1586,
        'es': 13,  # We would need about 20 shards for 1B sentences, but there are only 13
    }[language]
    ccnet_filepaths = [ccnet_dir / f'{language}_head_{i:04d}.json.gz' for i in range(n_shards)]
    raw_original_dir = dataset_dir / 'raw_original'
    raw_original_dir.mkdir(exist_ok=True, parents=True)
    output_dirs = [raw_original_dir / f'{language}_head_{i:04d}' for i in range(n_shards)]
    n_docs_per_file = 50000
    executor = get_executor(cluster=cluster, slurm_partition='debug', timeout_min=1 * 30, slurm_array_parallelism=16)
    jobs = []
    with executor.batch():
        for ccnet_filepath, output_dir in zip(ccnet_filepaths, output_dirs):
            if output_dir.exists():
                continue
            job = executor.submit(split_ccnet_shard, ccnet_filepath, output_dir, n_docs_per_file)
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]  # Wait for the jobs to finish

# Sentence tokenization
with log_action('Tokenizing sentences'):
    executor = get_executor(
        cluster=cluster,
        slurm_partition=slurm_partition,
        timeout_min=2 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
    )
    subshard_paths = get_subshard_paths(raw_original_dir)
    jobs = []
    with executor.batch():
        for i, subshard_path in enumerate(subshard_paths):
            sentences_path = dataset_dir / 'sentences' / f'{i:06d}.txt.gz'
            if sentences_path.exists():
                continue
            sentences_path.parent.mkdir(exist_ok=True, parents=True)
            # Should take a bit less than 10 minutes each
            job = executor.submit(sentence_tokenize_subshard, subshard_path, sentences_path, language)
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

embeddings_type_name = f'laser_{language}'
get_embeddings = lambda sentences: get_laser_embeddings(
    sentences, max_tokens=1000, language=language, n_encoding_jobs=8
)  # noqa: E731

# Create base index
with log_action('Creating base index'):
    n_train_sentences = 4e6
    train_sentences = []
    for sentences_path in get_sentences_paths(dataset_dir):
        for sentence in yield_lines(sentences_path):
            train_sentences.append(sentence)
            if len(train_sentences) == n_train_sentences:
                break
        if len(train_sentences) == n_train_sentences:
            break

    base_index_dir = dataset_dir / f'base_indexes/'
    base_index_dir.mkdir(exist_ok=True, parents=True)
    print(len(train_sentences))
    # This can be very long
    base_index_path = create_base_index(
        train_sentences, get_index_name(), get_embeddings, faiss.METRIC_L2, base_index_dir
    )

def sync_jobs(jobs):
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]    

# Compute embeddings
with log_action('Computing embeddings'):
    cache_dir = get_cache_dir(dataset_dir) / embeddings_type_name
    indexes_dir = cache_dir / 'indexes' / f'base-index-{get_file_hash(base_index_path)}'
    indexes_dir.mkdir(exist_ok=True, parents=True)
    db_sentences_paths = get_sentences_paths(dataset_dir)
    query_sentences_paths = db_sentences_paths
    executor = get_executor(
        cluster=cluster,
        slurm_partition=slurm_partition,
        timeout_min=4 * 60,
        slurm_array_parallelism=8,
    )

    processed = 0
    for i, sentences_path in enumerate(set(query_sentences_paths + db_sentences_paths)):
        #print(i)
        if get_index_path(sentences_path, indexes_dir).exists():
            continue
        # Should take about 30 minutes each
        with executor.batch():            
            job = executor.submit(
                compute_and_save_embeddings, sentences_path, base_index_path, get_embeddings, indexes_dir=indexes_dir
            )
        jobs.append(job)
        if len(jobs)==12 or i == len(db_sentences_paths)-1:
            sync_jobs(jobs)
            processed += len(jobs)
            print(processed)
            jobs=[]
    if jobs:
        sync_jobs(jobs)
        processed += len(jobs)
        print(processed)

# Mine the paraphrases
with log_action('Mining paraphrases'):
    nn_search_results_dir = cache_dir / 'nn_search_results'
    nn_search_results_dir.mkdir(exist_ok=True, parents=True)
    topk = 8
    nprobe = 16
    executor = get_executor(
        cluster=cluster,
        slurm_partition=slurm_partition,
        timeout_min=4 * 60,
        slurm_array_parallelism=1,
    )
    # Run NN search query file by query file
    processed = 0
    jobs = []
    for i, query_sentences_path in enumerate(tqdm(query_sentences_paths, desc='submitting queries')):
        print(f'query:{i}')
        if get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir).exists():
            continue
        # Should take about ~1h30 each
        with executor.batch():
            job = executor.submit(
                compute_and_save_nn_batched,
                query_sentences_path,
                db_sentences_paths,
                topk,
                nprobe,
                indexes_dir,
                nn_search_results_dir,
                delete_intermediary=True,
            )
        jobs.append(job)
        if len(jobs)==1 or i == len(query_sentences_paths)-1:
            sync_jobs(jobs)
            processed += len(jobs)
            print(f'queries_processed:{processed}')
            jobs=[]
    if jobs:
        sync_jobs(jobs)
        processed += len(jobs)
        print(f'queries_processed:{processed}')

# Filter candidate paraphrases
with log_action('Filtering candidate paraphrases'):
    pairs_dir = cache_dir / 'pairs'
    pairs_dir.mkdir(exist_ok=True, parents=True)
    filter_kwargs = {
        'density': 0.6,
        'distance': 0.05,
        'levenshtein': 0.2,
        'simplicity': 0.0,
        'filter_ne': False,
    }  # Best for paraphrases
    jobs = []
    paraphrase_pairs = []
    i = 0
    is_simpler = lambda pair: True  # noqa: E731
    # Only used when mining simplifications
    if filter_kwargs.get('simplicity', 0) > 0:
        while len(paraphrase_pairs) < 10000:
            paraphrase_pairs.extend(
                get_paraphrase_pairs(
                    query_sentences_paths[i],
                    db_sentences_paths,
                    base_index_path,
                    get_embeddings,
                    cache_dir,
                    topk,
                    nprobe,
                    filter_kwargs,
                )
            )
            i += 1
        simplicity_scorer = SimplicityScorer(language=language)
        simplicity_scorer.fit(paraphrase_pairs)
        is_simpler = lambda pair: simplicity_scorer.score(*pair) > filter_kwargs['simplicity']  # noqa: E731
    executor = get_executor(
        cluster=cluster,
        slurm_partition=slurm_partition,
        timeout_min=4 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
    )
    processed = 0
    for i, query_sentences_path in enumerate(tqdm(query_sentences_paths, desc='query')):
        simplification_pairs_path = get_pairs_path(
            query_sentences_path, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
        )
        if simplification_pairs_path.exists():
            continue
        with executor.batch():
            # Should take ~10 minutes
            job = executor.submit(
                compute_and_save_simplification_pairs,
                query_sentences_path=query_sentences_path,
                db_sentences_paths=db_sentences_paths,
                base_index_path=base_index_path,
                cache_dir=cache_dir,
                pairs_dir=pairs_dir,
                get_embeddings=get_embeddings,
                topk=topk,
                nprobe=nprobe,
                language=language,
                filter_kwargs=filter_kwargs,
                is_simpler=is_simpler,
            )
        jobs.append(job)
        if len(jobs)==1 or i == len(query_sentences_paths)-1:
            sync_jobs(jobs)
            processed += len(jobs)
            print(f'queries_processed:{processed}')
            jobs=[]
    if jobs:
        sync_jobs(jobs)
        processed += len(jobs)
        print(f'queries_processed:{processed}')

with log_action('Wrapping up paraphrases'):
    simplification_pairs = get_simplification_pairs_paths(
        query_sentences_paths, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
    )
    results_str = f'query-{get_files_hash(query_sentences_paths)}_db-{get_files_hash(db_sentences_paths)}_topk-{topk}_nprobe-{nprobe}'
    filter_str = get_filter_string_representation(filter_kwargs)
    dataset = f'uts_{language}_{results_str}_{filter_str}'
    print(combine_simplifications_in_dataset(simplification_pairs, dataset))
