from sys import stderr
import json
from pprint import pprint

dir = 'result/'
errors = {}
sep = ' & '
end = r' \\' + '\n'
dataset_start = r'\midrule' + '\n'

file_for_xp = {
    'EDA': '_out_eda_{dataset}-metrics.jsonl',
    'AEDA': '_out_aeda_{dataset}-metrics.jsonl',
    'RTT': '_out_translation_{dataset}-metrics.jsonl',
    'BART-dbs': '_out_bart-dbs-uni-flat-1.0_{dataset}-metrics.jsonl',
    'Ours': '{dataset}-back-translations_3kw_5paraphrases-p1.0syn-p0.5shuf_checkpoint-65536-greedy-metrics.jsonl',
}


def parse(str, sep=','):
    return [e.strip() for e in str.split(sep)]


first_header = ''
_sum = {}
datasets = ['BANKING77', 'HWU64', 'OOS', 'Liu']
for dataset in datasets:
    for xp, file_in in file_for_xp.items():
        file_in = file_in.format(dataset=dataset)
        printable_dataset = dataset.replace('OOS', 'Clinic150')
        try:
            with open(dir + file_in, encoding="utf-8") as f:
                metrics = json.load(f)
            if not first_header:
                first_header = metrics['debug']['table_header']
                printable_header = [r'\textbf{' + e + '}' for e in ['Dataset', 'Method'] + parse(first_header)]
                print(*printable_header, sep=sep, end=end)
                print(dataset_start, end='')
            else:
                assert metrics['debug']['table_header'] == first_header
            print(printable_dataset, xp, *parse(metrics['debug']['table_values']), sep=sep, end=end)

            if xp not in _sum:
                _sum[xp] = [0 for i in range(len(parse(first_header)))]
            for i, value in enumerate(parse(metrics['debug']['table_values'])):
                mean = value.split('±')[0].split('×÷')[0].strip()
                _sum[xp][i] += float(mean)

        except FileNotFoundError as e:
            print(printable_dataset, xp, *['-' for i in range(len(parse(first_header)))], sep=sep, end=end)
            errors[dir + file_in] = e
        if 'nan' in metrics['debug']['table_values']:
            errors[dir + file_in + ' warning'] = 'nan present in row'

    print(dataset_start, end='')

for xp in _sum:
    print('average', xp, *[f'{s / len(datasets):.3g}' for s in _sum[xp]], sep=sep, end=end)

print('Errors:', file=stderr)
pprint(errors, stream=stderr)
