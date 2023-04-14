"""
usage examle:
python logs_to_csv.py xps/*/protaugment/metrics.json
"""

import os, sys
import json
import argparse
import re

import numpy as np


def get_score_from_metrics_fp(metrics_fp):
    with open(os.path.join(metrics_fp), "r") as _f:
        metrics = json.load(_f)

    best_valid_acc = 0.0
    test_acc = 0.0
    assert len(metrics["valid"]) == len(metrics["test"])
    for valid_episode, test_episode in zip(metrics["valid"], metrics["test"]):
        for valid_metric_dict in valid_episode["metrics"]:
            if valid_metric_dict["tag"].startswith("acc"):
                valid_acc = valid_metric_dict["value"]
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    for test_metric_dict in test_episode["metrics"]:
                        if test_metric_dict["tag"].startswith("acc"):
                            test_acc = test_metric_dict["value"]
                        break
                    else:
                        raise ValueError
                break
        else:
            raise ValueError
    return best_valid_acc, test_acc


def print_scores(valids, tests, path, verbose=False, format='.3f'):
    print(
        f'{np.mean(valids):{format}}±{np.std(valids):{format}}',
        f'{np.mean(tests):{format}}±{np.std(tests):{format}}',
        len(valids),
        path
    )
    if verbose >= 2:
        print(' ', valids, tests)


def main(args):
    scores_valid = {}
    scores_test = {}
    errors = []
    for path in args.log_files:  # in sorted(args.log_files, key=lambda x: os.path.getmtime(x)):
        if args.verbose >= 1:
            print('reading', path)
        try:
            valid, test = get_score_from_metrics_fp(path)
            path = re.sub(r'(-cv[0-9]*)', '', path)  # group different crossvalidation runs together
            if not args.keep_xp_id:
                path = '_'.join(path.split('_')[1:])  # group different xp folders, but loose xp identifier
            if args.fuse_datasets:
                path = path.replace('_BANKING77', '').replace('_HWU64', '').replace('_Liu', '').replace('_OOS', '')  # group different datasets togethers
            if path not in scores_valid:
                scores_valid[path] = []
                scores_test[path] = []
            scores_valid[path].append(valid)
            scores_test[path].append(test)
        except UnicodeDecodeError as e:
            errors.append(f"could not read {path} : {e}")

    if errors:
        print('---')
        print('!!! errors : !!!', *errors, sep='\n')

    print(f'--- valid, test, n_xps, xps{", valids, tests" if args.verbose else ""} ---')
    for path in scores_valid:
        print_scores(scores_valid[path], scores_test[path], path, args.verbose)

    print('--- best test acc ---')
    path = max(scores_test, key=lambda k: sum(scores_test[k])/len(scores_test[k]))
    print_scores(scores_valid[path], scores_test[path], path, args.verbose)

    print('--- best valid acc ---')
    path = max(scores_valid, key=lambda k: sum(scores_valid[k])/len(scores_valid[k]))
    print_scores(scores_valid[path], scores_test[path], path, args.verbose)

    return (scores_valid, scores_test)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', nargs='+', type=str)
    parser.add_argument('-v', '--verbose', type=int, default=1, choices=[0,1,2])
    parser.add_argument('-i', '--keep_xp_id', action='store_true')
    parser.add_argument('-d', '--fuse_datasets', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
