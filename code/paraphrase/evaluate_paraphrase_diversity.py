#!/home/chr06111/NAG/ProtAugment/venv-protaugment/bin/python3
import sys; sys.path.insert(0, '/home/chr06111/NAG/ProtAugment')
## \-> use ProtAugment environment and python path


## parse command-line arguments
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--top", action='store_true')
    parser.add_argument("--cached", action='store_true')
    parser.add_argument("--light", action='store_true')
    args = parser.parse_args()

    do_heavy_metrics = not args.light
else:
    do_heavy_metrics = True

## imports
import os
from os import path
import json, pickle
from typing import List
from itertools import islice, chain
from functools import lru_cache
from dataclasses import dataclass
import math

import numpy as np

def cache_path(in_path):
    return in_path.rstrip('.jsonl') + '-metrics.pickle'

if not (args.cached or args.top) or not path.isfile(cache_path(args.in_file)):
    from sacrebleu import corpus_bleu
    from tqdm import tqdm
    import sacrebleu
    from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
    import torch

    print('Loading tensorflow, pytorch and transformers modules...')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # less tensorflow stdout clutter. 0: debug, 1: info, 2: warning, 3: error
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')  # prevent tf from reserving all the gpu memory,
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # so pytorch can have some
    if do_heavy_metrics:
        from transformers import BertForMaskedLM, BertTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast
        from bleurt.score import BleurtScorer

    from models.use import use_embedder
    from utils.data import get_jsonl_data

    ## init
    if do_heavy_metrics \
            and 'model_name' not in globals():  # enables faster module reloads by bypassing this section
        device = "cuda"
        model_name = 'bert-base-uncased'
        print(f'Loading', model_name, 'model...')
        bert_model = BertForMaskedLM.from_pretrained(model_name).to(device)
        bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)

        model_name = "distilgpt2"
        print(f'Loading', model_name, 'model...')
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

        # model_name = "../BLEURT-20"  # full model  # cd .. && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip && unzip BLEURT-20.zip && rm BLEURT-20.zip && cd paraphrase
        # model_name = "../BLEURT-20-D12"  # biggest distilled model  # cd .. && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip && unzip BLEURT-20-D12.zip && rm BLEURT-20-D12.zip && cd paraphrase
        model_name = "../BLEURT-20-D3"  # smallest distilled model  # cd .. && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip && unzip BLEURT-20-D3.zip && rm BLEURT-20-D3.zip && cd paraphrase
        print(f'Loading', model_name, 'model...')
        bleurt_scorer = BleurtScorer(model_name)


## metrics
def dist_k(texts: List[str], k: int, lowercase: bool = False) -> float:
    if lowercase:
        texts = [t.lower() for t in texts]
    splitted = [
        t.strip().split()
        for t in texts
    ]
    k_grams = [
        tuple(s[i:i + k])
        for s in splitted for i in range(0, len(s) - k + 1) if len(s) >= k
    ]
    n_distinct_k_grams = len(set(k_grams))
    n_tokens = sum([len(s) for s in splitted])
    return n_distinct_k_grams / n_tokens


def get_metrics(sentence, references, vs_source=False):
    metrics = {}

    if vs_source:
        assert len(references) == 1
        metrics["len_out/len_in"] = len(sentence) / len(references[0])

    # BLEU
    metrics["bleu"] = sacrebleu.sentence_bleu(sentence, references, smooth_method='exp').score

    # cosine dist between USEmbedding
    sentence_use = use_embedder.embed_many([sentence])
    reference_uses = use_embedder.embed_many(references)
    metrics["use_similarity"] = np.mean(cosine_similarity(sentence_use, reference_uses))

    # dist-k
    for k in (2, 3):
        metrics[f"dist-{k}"] = dist_k([sentence] + references, k=k, lowercase=True)

    # --- heavier metrics ---
    if not do_heavy_metrics:
        return metrics

    # BLEURT
    metrics["bleurt"] = np.mean(bleurt_scorer.score(candidates=[sentence] * len(references), references=references))

    return metrics


def get_single_metrics(sentence):
    metrics = {}

    metrics["n_chars"] = len(sentence)

    # --- heavier metrics ---
    if not do_heavy_metrics:
        return metrics

    import torch
    # BERT perplexity
    tensor_input = bert_tokenizer.encode(sentence, return_tensors='pt').to(device)
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1, device=device).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, bert_tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != bert_tokenizer.mask_token_id, -100)
    # since a 512 tokens input would lead to a batch of
    with torch.no_grad():
        try:
            loss = bert_model(masked_input, labels=labels).loss
            metrics["bert_perplexity"] = np.exp(loss.item())
        except RuntimeError as e:  # when CUDA out of memory (TODO: split big batchs)
            print(e)
            print(f"provoqued by input sentence '{sentence}'")
            metrics["bert_perplexity"] = None

    # GPT2 perplexity (TODO: one-token case)
    encodings = gpt2_tokenizer(sentence, return_tensors="pt").to(device)
    max_length = gpt2_model.config.n_positions
    stride = 512
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    metrics["gpt2_perplexity"] = torch.exp(torch.stack(nlls).sum() / end_loc).detach().cpu().numpy().item()

    return metrics


## run, save and load metrics
def compute(in_file, cache_file=None):
    data = get_jsonl_data(in_file)
    n = len(data[0]["tgt_texts"])
    for d in data:
        assert len(d["tgt_texts"]) == n, "cannot evaluate paraphrase diversity if number of paraphrases vary"

    metrics_vs_source = []
    metrics_vs_minibatch = []
    single_metrics = []
    for d in tqdm(data, desc='Computing metrics'):
        source_sentence = d["src_text"]
        paraphrases = d["tgt_texts"]
        for paraphrase in paraphrases:
            metrics_vs_source.append(get_metrics(paraphrase, [source_sentence], vs_source=True))
            metrics_vs_minibatch.append(get_metrics(paraphrase, paraphrases))
            single_metrics.append(get_single_metrics(paraphrase))

    results = (data, metrics_vs_source, metrics_vs_minibatch, single_metrics)
    if cache_file is not None:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return results


def load(cache_file):
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def load_or_compute(in_file):
    cache_file = cache_path(in_file)
    if path.isfile(cache_file):
        print(f"Loading cached file '{cache_file}'")
        return load(cache_file)
    else:
        return compute(in_file, cache_file)


def to_dict(results):
    (data, metrics_vs_source, metrics_vs_minibatch, single_metrics) = results
    breakpoint()


## main for aggregates on metrics
def benchmark(in_file, out_file, cached=False, print_precision=3, add_debug=True):
    if cached:
        (data, metrics_vs_source, metrics_vs_minibatch, single_metrics) = load_or_compute(in_file)
    else:
        (data, metrics_vs_source, metrics_vs_minibatch, single_metrics) = compute(in_file, cache_path(in_file))

    aggregates = {}

    if add_debug:
        debug = {'single metrics': '', 'metrics vs. source': '', 'metrics vs. minibatch': '',
                 'table_header': '', 'table_values': ''}
        aggregates['debug'] = debug
        table_metrics = {
            'single metrics': ['n_chars', 'gpt2_perplexity', 'bert_perplexity'],
            'metrics vs. source': ['use_similarity'],
            'metrics vs. minibatch': ['dist-2'],
        }

    for metric_type, metrics in {
        'single metrics': single_metrics,
        'metrics vs. source': metrics_vs_source,
        'metrics vs. minibatch': metrics_vs_minibatch,
    }.items():
        aggregate = {}
        for metric_name in metrics[0].keys():
            values = [metric[metric_name] for metric in metrics]
            if None in values or any(np.isnan(values)):
                paraphrases = [p for d in data for p in d['tgt_texts']]
                ko = {p: v for (p, v) in zip(paraphrases, values) if v is None or np.isnan(v)}
                print(r'/!\ Warning: ignored', len(ko), 'sentences with', metric_name, 'None/NaN from', in_file)
                print('detail:', ko)
                values = [v for v in values if v is not None and not np.isnan(v)]
            if not 'perplexity' in metric_name and metric_name != 'len_out/len_in':
                mean = np.mean(values)
                pstd = np.std(values)
                aggregate[metric_name] = {'mean': float(mean), 'pstd': float(pstd)}
                symbol = '±'
            else:
                mean = np.exp(np.mean(np.log(values)))
                pstd = np.exp(np.std(np.log(values)))
                aggregate[metric_name] = {'geo mean': float(mean), 'geo pstd': float(pstd)}
                symbol = '×÷'
            # if str(mean) == 'nan': breakpoint()

            if add_debug:
                debug[metric_type] += f'{metric_name}: {mean:.{print_precision}g} {symbol} {pstd:.{print_precision}g},  '
                if metric_name in table_metrics[metric_type]:
                    suffix = {'metrics vs. source': ' vs src', 'metrics vs. minibatch': ' vs batch', 'single metrics': ''}
                    debug['table_header'] += f'{metric_name}{suffix[metric_type]},  '
                    debug['table_values'] += f'{mean:.{print_precision}g} {symbol} {pstd:.{print_precision}g},  '
        if add_debug:
            debug[metric_type] = debug[metric_type][:-3]

        aggregates[metric_type] = aggregate
    aggregates['n_paraphrases'] = len(values)

    if add_debug:
        debug['table_header'] = debug['table_header'][:-3]
        debug['table_values'] = debug['table_values'][:-3]
        print(
            '-- metrics summary --',
            '\n'.join(f"{k}: {{{debug[k]}}}" for k in ['single metrics', 'metrics vs. source', 'metrics vs. minibatch']),
            f'n_paraphrases: {aggregates["n_paraphrases"]}',
            '-- paper table --',
            debug['table_header'],
            debug['table_values'],
        sep='\n')

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as file:
        json.dump(aggregates, file, indent=1, ensure_ascii=False)


## main for extracting paraphrases according to metrics percentiles
def top(in_file, n_samples=10, percentiles=[1, 25, 50, 75, 99], alway_display_source=True, debug=False):
    (data, metrics_vs_source, metrics_vs_minibatch, single_metrics) = load_or_compute(in_file)

    sort_metrics = {
        'single metrics': ['gpt2_perplexity'],
        'metrics vs. source': ['use_similarity'],
        'metrics vs. minibatch': ['dist-2'],
    }
    printable_type = {
        'single metrics': '',
        'metrics vs. source': ' vs. source',
        'metrics vs. minibatch': ' vs. minibatch',
    }
    sources = {p: d['src_text'] for d in data for p in d['tgt_texts']}
    batchs = {p: d['tgt_texts'] for d in data for p in d['tgt_texts']}

    if debug: ex_for_percentile_for_metric_for_type = {}
    for metric_type, metrics in {
        'single metrics': single_metrics,
        'metrics vs. source': metrics_vs_source,
        'metrics vs. minibatch': metrics_vs_minibatch,
    }.items():
        if debug: ex_for_percentile_for_metric = {}
        for metric_name in metrics[0].keys():
            if metric_name not in sort_metrics[metric_type]:
                continue
            print('\n\n=====', metric_name + printable_type[metric_type], '=====')
            paraphrases = [p for d in data for p in d['tgt_texts']]
            values = [metric[metric_name] for metric in metrics]
            if None in values or any(np.isnan(values)):
                ko = {p: v for (p, v) in zip(paraphrases, values) if v is None or np.isnan(v)}
                print(r'/!\ Warning: ignored', len(ko), 'sentences with', metric_name, 'None/NaN from', in_file)
                print('detail:', ko)
                values = [v for v in values if v is not None and not np.isnan(v)]
            reverse = 'perplexity' not in metric_name
            value_for_paraphrase = {p: v for v, p in zip(values, paraphrases)}  # deduplication
            paraphrases_and_values = [(p, value_for_paraphrase[p]) for p in sorted(
                value_for_paraphrase, key=value_for_paraphrase.get, reverse=reverse)]  # sort
            ex_for_percentile = {}
            for percentile in percentiles:
                print(f'\n=== {n_samples} samples around top{percentile}% === ')
                len_data = len(paraphrases_and_values)
                i_min = len_data * percentile*0.01 - n_samples / 2
                i_min = max(min(math.floor(i_min), len_data - n_samples), 0)
                i_max = min(i_min + n_samples, len_data)
                padding = len(str(len_data))
                for i, (p, v) in enumerate(paraphrases_and_values[i_min:i_max]):
                    header = f'rank: {i_min+i+1:{padding}}/{len_data}, score: {f"{v:.3g},":<8}'
                    print(header, f'paraphrase: "{p}"')
                    if metric_type == 'metrics vs. source' or alway_display_source:
                        print(' '*len(header), f'source: "{sources[p]}"')
                    if metric_type == 'metrics vs. minibatch':
                        for b in batchs[p]:
                            print(' '*len(header), f'batch: "{b}"')
                if debug: ex_for_percentile[percentile] = paraphrases_and_values[i_min:i_max]
            if debug: ex_for_percentile_for_metric[metric] = ex_for_percentile
        if debug: ex_for_percentile_for_metric_for_type[metric_type] = ex_for_percentile_for_metric
    if debug: return ex_for_percentile_for_metric_for_type


if __name__ == "__main__":
    if args.top:
        """usage: ./evaluate_paraphrase_diversity.py --top --in_file xps/057_aug_BANKING77-limit1000_0.5syn-0.25shuf_ckpt-65536/paraphrases.jsonl"""
        top(args.in_file, n_samples=10, percentiles=[1, 25, 50, 75, 99])
        # top(args.in_file, n_samples=1, percentiles=[0, 1, 25, 50, 75, 99, 100])
    else:
        """usage: ./evaluate_paraphrase_diversity.py --in_file xps/057_aug_BANKING77-limit1000_0.5syn-0.25shuf_ckpt-65536/paraphrases.jsonl"""
        args.out_file = args.out_file or args.in_file.rstrip('.jsonl') + '-metrics.jsonl'
        benchmark(args.in_file, args.out_file, args.cached)
