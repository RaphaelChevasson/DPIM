import sys, json, argparse

sys.path.insert(0, '../aeda_nlp/code')
from eda import eda
from aeda import insert_punctuation_marks as aeda

# eda and aeda inference

def main(in_file, eda_out_file, aeda_out_file, n_augs=9, limit=None, verbose=False):

    with open(in_file, encoding='utf-8') as f_in, \
            open(eda_out_file, 'w', encoding='utf-8') as f_eda, \
            open(aeda_out_file, 'w', encoding='utf-8') as f_aeda:

        for i, line in enumerate(f_in):
            if limit is not None and i > limit:
                break

            if in_file.endswith('.jsonl'):
                line = json.loads(line)['src_text']
            if line == '':
                pass

            try:
                eda_paraphrases = eda(line, num_aug=n_augs)[:-1]  # eda adds the source at end, remove it
            except Exception as e:
                print(f"sentence '{line}' at line {i} of file '{in_file}' provoked eda error '{e}'; "
                      f"repeating source instead")
                eda_paraphrases = [line for i in range(n_augs)]
            aeda_paraphrases = [aeda(line) for i in range(n_augs)]

            if verbose:
                print(f'--- {i} ---',
                      line,
                      f'eda: {eda_paraphrases}',
                      f'aeda: {aeda_paraphrases}', sep='\n')
#            f_sources.write(line + '\n')

            if eda_out_file.endswith('.jsonl'):
                f_eda.write(json.dumps({'src_text': line, 'tgt_texts': eda_paraphrases}) + '\n')
            else:
                f_eda.writelines(eda_paraphrases + '\n')

            if aeda_out_file.endswith('.jsonl'):
                f_aeda.write(json.dumps({'src_text': line, 'tgt_texts': aeda_paraphrases}) + '\n')
            else:
                f_aeda.writelines(aeda_paraphrases + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--eda_out_file', type=str, default=None)
    parser.add_argument('--aeda_out_file', type=str, default=None)
    parser.add_argument('--n_augs', type=int, default=9)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    if args.in_file:
        main(**args)
    else:
        for dataset in ('BANKING77', 'HWU64', 'OOS', 'Liu'):
            in_file = f'../ProtAugment/data/{dataset}/back-translations.jsonl'
            main(
                in_file,  # limit=3, verbose=True,
                eda_out_file=f'../aeda_nlp/_out_eda_{dataset}.jsonl',
                aeda_out_file=f'../aeda_nlp/_out_aeda_{dataset}.jsonl'
            )

# training: TODO
