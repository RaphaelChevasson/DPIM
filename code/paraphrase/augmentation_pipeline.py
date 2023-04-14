#from nlpretext.augmentation.text_augmentation import augment_text
#from POINTER.utils_ig import house_robber
#print(house_robber([100, 3, 4, 5, 6, 0, 0, 5, 0, 10, 5], skip=2))  # return the right indexes, exept index 0


print('=== importing modules ===')

import sys, os, contextlib, json
from shutil import rmtree
import argparse
import random
from math import ceil, factorial
from itertools import chain, repeat, permutations, product
from dataclasses import dataclass, fields, asdict as dataclass_as_dict
from typing import List

from tqdm import tqdm


print('=== parsing arguments ===')

def str2bool(v):
    """Used as an argparse type to enables parsing of positional requiered booleans"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
# === flow parameters ===
parser.add_argument('do_paraphrase',   type=str2bool, default=False)
parser.add_argument('do_pointer',      type=str2bool, default=False)
parser.add_argument('do_protaugment',  type=str2bool, default=True)
# === I/O parameters 1/2 ===
parser.add_argument('dataset',         type=str, choices=('BANKING77', 'HWU64', 'OOS', 'Liu'))
parser.add_argument('--split_into',    type=int, default=1)
parser.add_argument('--split_index',   type=int, default=0)
parser.add_argument('--limit_dataset', type=int, default=False)
parser.add_argument('--in_file',       type=str, default=None)
parser.add_argument('--in_path',       type=str, default=None)
parser.add_argument('--log_path',      type=str, default='../paraphrase/logs')
parser.add_argument('--tmp_path',      type=str, default='../paraphrase/tmp')
parser.add_argument('--out_path',      type=str, default='../paraphrase/result')
# === paraphrases parameters 1/2 ===
parser.add_argument('--paraphrases_path',              type=str,      default='../paraphrase')
parser.add_argument('--baseline',                      type=str,      default=None) # 'junk', 'copy', 'backtranslation'/others paired with --in_path
parser.add_argument('--n_outputs',                     type=int,      default=5)
parser.add_argument('--n_keywords',                    type=int,      default=3)
parser.add_argument('--filter_min_n_keyword s',        type=int,      default=0)  # 0 is disabled, don't filter
parser.add_argument('--pick_longest_until_n_keywords', type=str2bool, default=True)  # if less than n kw extracted, pick longest words until n extracted
parser.add_argument('--p_synonyms',                    type=float,    default=0.5)
parser.add_argument('--synonyms_method',               type=str,      default='ewiser', choices=('ewiser', 'wordnet lesk'))
parser.add_argument('--conjugate_synonyms',            type=str2bool, default=True)
parser.add_argument('--p_shuffles',                    type=float,    default=1)
parser.add_argument('--keep_punctuation',              type=str2bool, default=False)
parser.add_argument('--do_diversity_eval',             type=str2bool, default=None)
# === POINTER parameters 1/2 ===
parser.add_argument('--pointer_model',     type=str, default=None)
# === ProtAugment parameters 1/2 ===
parser.add_argument('--model_to_finetune', type=str, default=None)
parser.add_argument('--CV',                type=int, nargs="+", default=[1])  # list(range(5))
parser.add_argument('--C',                 type=int, nargs="+", default=[5])
parser.add_argument('--K',                 type=int, nargs="+", default=[5])  # [1, 5]
parser.add_argument('--regimes',           type=str, nargs="+", default=['full'])  # ['10samp', 'full']
parser.add_argument('--protaugment_seed',  type=int, default=42)


args = parser.parse_args()
with open(f'{args.log_path}/config.log', 'w', encoding='utf-8') as f:
    for arg in vars(args):
        locals()[arg] = getattr(args, arg)
        print(f'{arg}={locals()[arg]}')
        print(f'{arg}={locals()[arg]}', file=f)


# === paraphrases parameters 2/2 ===

#paraphrases_path = '../paraphrase'
#paraphrases_env = f'{paraphrases_path}/venv-paraphrase/bin/python'
# \-> TODO, launch this script with ewiser venv in the meantime
if do_diversity_eval is None:
    do_diversity_eval = do_pointer or baseline

# === POINTER parameters 2/2 ===

pointer_path = '../POINTER'
pointer_env = f'{pointer_path}/venv-POINTER/bin/python'
if pointer_model is None:
    pointer_model = f'{pointer_path}/ckpt/assistant_model_maison/checkpoint-20000'  # [...]/wiki_model, news_model, assistant_model_maison/checkpoint-20000
sample = 'greedy'  # 'greedy', 'sample'  # TODO: implement sample
# TODO: implement n_namples
no_ins_at_end = 'when_punctuation_at_end' if keep_punctuation else 'false'


# === ProtAugment parameters 2/2 ===

protaugment_path = '../ProtAugment'
protaugment_env = f'{protaugment_path}/venv-protaugment/bin/python'
if model_to_finetune is None:
    model_to_finetune = f"{protaugment_path}/transformer_models/{dataset}/fine-tuned"


# === I/O preparation ===

os.makedirs(f'{log_path}', exist_ok=True)
os.makedirs(f'{tmp_path}', exist_ok=True)
os.makedirs(f'{out_path}', exist_ok=True)

# === I/O parameters 2/2 ===

do_in_from_file = True
if in_path is None:
    in_path = f'../ProtAugment/data/{dataset}'
if in_file is None:
    # in_file can be sys.stdin or a file path
    in_file = f'{in_path}/back-translations.jsonl'
    # in_file = f'{pointer_path}/data/{dataset}_valid.txt'
inputs = [  # used when do_in_from_file == False
    "I think my account has been hacked there are charges on there I don't recognize.",
    "can you make a red shade of light in the living room"
]
cache_keyword_extaction = True  # reuse last results, useful when debugging and it has already be done one time
cache_synonyms = True

desc_inputs =        f'{dataset}-{".".join(in_file.split("/")[-1].split(".")[:-1])}' \
                       f'{f"-limit{limit_dataset}" if limit_dataset else ""}' \
                       f'{f"-split{split_index}of{split_into}" if split_into>1 else ""}'
desc_kw_extration =  f'{desc_inputs}_{n_keywords}kw'
desc_synonyms =      f'{desc_inputs}_{synonyms_method}'
desc_augmentations = f'{desc_kw_extration}_{n_outputs}paraphrases' \
                       f'-p{p_synonyms}syn-p{p_shuffles}shuf'
desc_pointer =       f'{desc_augmentations}_{pointer_model.split("/")[-1]}-{sample}'
desc_baseline =      f'{desc_inputs}_baseline-{n_outputs}-{baseline}'
desc_paraphrase = desc_pointer if not baseline else desc_baseline

desc_protaugment =   f'lm-{os.path.split(model_to_finetune)[-1]}'

if 'paraphrases.jsonl' not in os.listdir(out_path):  # one dir per xp
    out_file_base = f'{out_path}/paraphrases'
else:                                                # one shared dir
    out_file_base = f'{out_path}/{desc_paraphrase}'
do_out_to_file = True
out_file_txt = out_file_base + '.txt'  # sys.stdout or a file path
out_file_jsonl = out_file_base + '.jsonl'
verbose = True
verbose_file = out_file_base + '_verbose.txt'  # sys.stdin or a file path
print_outputs_after = False  # debug pointer paraphrase after the augmentation tree rather than inside it

csv_sep = ','


# === temporary files ===

kw_extraction_in_file_name =  f'{tmp_path}/kw_extraction_in-{desc_inputs}.txt'
kw_extraction_out_file_name = f'{tmp_path}/kw_extraction_out-{desc_kw_extration}.txt'
synonyms_out_file_name =      f'{tmp_path}/synonyms_out-{desc_synonyms}.json'
agmentations_out_file_name =  f'{tmp_path}/augmentations_out-{desc_augmentations}.json'
pointer_in_file_name =        f'{tmp_path}/pointer_in-{desc_augmentations}.txt'
pointer_out_file_name =       f'{tmp_path}/pointer_out-{desc_pointer}.txt'


print('=== importing additional modules ===')

if do_paraphrase:
    import numpy as np

    print('nltk...'); from nltk.wsd import lesk
    
    print('pointer...'); import keyword_extraction  # from POINTER, please add ../POINTER to your $PYTHONPATH
    print('ewiser...'); from ewiser_synonyms import get_tokenizer as get_ewiser_tokenizer

    keywords_tokenizer = get_ewiser_tokenizer()  # TODO: check same in kw extraction and synonyms
    # TODO: errr yake use segtok.tokenizer, POINTER's yake then check they are in bert_uncased tokenizer vocab,
    #  I did lesk with nltk, and ewiser use spacy...

    if synonyms_method == "ewiser" and not cache_synonyms or not os.path.isfile(synonyms_out_file_name):
        from ewiser.spacy import disambiguate
        from spacy import load

        if conjugate_synonyms:
            print('pattern.en...'); from pattern.en import tenses, conjugate
            try:
                conjugate('test')
            except (StopIteration, RuntimeError):
                raise ImportError("Please patch your pattern.en dependency:"
                                  " https://github.com/clips/pattern/issues/308#issuecomment-826404749"
                                  " or use --conjugate_synonyms False")


# === helper functions ===

@contextlib.contextmanager
def _open(filename_or_opened_file, *args, encoding='utf-8', **kwargs):
    """helper to handle both file paths and sys.std_*"""
    if filename_or_opened_file is None:
        yield os.devnull
    elif filename_or_opened_file in [sys.stdin, sys.stdout, sys.stderr]:
        yield filename_or_opened_file
    else:
        fh = open(filename_or_opened_file, *args, encoding=encoding, **kwargs)
        try:
            yield fh
        finally:
            fh.close()

def run_and_exit_if_fail(command):
    """call a shell commands synchronously, and exit on fail"""
    print('running command:', command)
    status = os.system(command)
    if status != os.EX_OK:
        sys.exit(status)

def gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[1:-1]  # [1:-1] removes header and last empty line
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values



# === main ===


print(f'=== reading inputs from {in_file} ===')

if do_in_from_file:
    with _open(in_file) as f:
        dataset_size = limit_dataset if limit_dataset else len(f.read().splitlines())
        f.seek(0)
        split_size = ceil(dataset_size / split_into)
        split = range(split_size*split_index, min(dataset_size, split_size*(split_index+1)))
        print(f'dataset size {dataset_size}, split {split.start}:{split.stop}')
        file_inputs = f.read().splitlines()[split.start:split.stop]
        if in_file.endswith('.jsonl'):
            inputs = [json.loads(line)['src_text'] for line in file_inputs]
    if baseline not in ['copy', "junk"]:
        with _open(in_file) as f:
            backtranslations = [json.loads(line) for line in file_inputs]
            backtranslations = {b['src_text']: b['tgt_texts'] for b in backtranslations}


print('=== storing inputs ===')

Keywords = List[str]
@dataclass
class Sentence:
    input: str = None
    keywords: Keywords = None
    keywords_synonyms: List[Keywords] = None
    keywords_shuffled: List[List[Keywords]] = None
    outputs: List[List[str]] = None

sentences = [Sentence(input=input) for input in inputs]


'''
in_file_jsonl = f"{pointer_path}/data/all_train.jsonl"
out_identity_file_jsonl = f"{pointer_path}/data/all_train.identity.jsonl"
if in_file_jsonl or out_identity_file_jsonl:
    print(f'=== writing formated inputs to {in_file_jsonl} and {out_identity_file_jsonl} ===')
    with _open(in_file_jsonl, 'w') as ff, _open(out_identity_file_jsonl, 'w') as fi:
        for s in tqdm(sentences, desc='Formatting'):
            input = s.input.rstrip('\r\n')
            json.dump({'src_text': input}, ff, ensure_ascii=False)
            print('', file=ff)  # '\r\n'
            json.dump({'src_text': input, 'tgt_texts': [input]}, fi, ensure_ascii=False)
            print('', file=fi)  # '\r\n'
    exit()
'''


if baseline:
    do_paraphrase = False
    do_pointer = False
    verbose = False
    if baseline == 'copy':
        for s in sentences:
            s.outputs = [s.input for i in range(n_outputs)]
    elif baseline == 'junk':
        for s in sentences:
            s.outputs = ['junk' for i in range(n_outputs)]
    else:
        for s in sentences:
            assert len(backtranslations[s.input]) >= n_outputs
            s.outputs = backtranslations[s.input][0:n_outputs]


elif do_paraphrase:
    #print('=== activating paraphrase virtual environment ===')
    #activate_script = paraphrases_env.rstrip('/python')+'/activate_this.py'
    #execfile(activate_script, dict(__file__=activate_script))


    print('=== extracting keywords using yake and some post-processing ===')

    if not (cache_keyword_extaction and os.path.isfile(kw_extraction_out_file_name)):
        with open(kw_extraction_in_file_name, 'w', encoding="utf-8") as f:
            for s in sentences:
                f.write(s.input+'\n')
        keyword_extraction.main(['--n_keys', str(n_keywords), '--file', kw_extraction_in_file_name])
        os.rename(kw_extraction_in_file_name[:-3] + 'key.txt', kw_extraction_out_file_name)
    else:
        print('loading cached results...')
    with open(kw_extraction_out_file_name, encoding="utf-8") as f:
        for s, line in zip(sentences, f.read().splitlines()):
            s.keywords = keywords_tokenizer(line)


    if pick_longest_until_n_keywords:
        print(f'=== picking longest words until we get {n_keywords} ===')

        from pytorch_transformers.tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab_list = list(tokenizer.vocab.keys())
        for s in sentences:
            if len(s.keywords) < n_keywords:
                unique_words = set(keywords_tokenizer(s.input))
                unique_words = [word for word in unique_words if word not in s.keywords]
                keywords = s.keywords + sorted(unique_words, key=len, reverse=True)[:n_keywords - len(s.keywords)]
                if len(keywords) < n_keywords:
                    print(f'Warning: sentence contains {len(keywords)} < {n_keywords} unique words, so '
                          f'it will only have {len(keywords)} keywords: "{s.input}"')
                keywords_original_order = []
                for k in keywords_tokenizer(s.input):
                    if k.lower() in [kw.lower() for kw in keywords]:
                        keywords_original_order.append(k.lower())
                s.keywords = keywords_original_order


    print('=== replacing by synonyms ===')

    if cache_synonyms and os.path.isfile(synonyms_out_file_name):
        print('loading cached results...')
        synonyms_method += "_cached"
        with open(synonyms_out_file_name, encoding="utf-8") as f:
            cached_synonyms = json.load(f)
        assert all((s.input in cached_synonyms for s in sentences)), f'{s.input} not in {synonyms_out_file_name}'
    elif synonyms_method == "ewiser":
        print('== loading ewiser ==')

        spacy_checkpoint = '../ewiser/ckpt/ewiser.semcor+wngt.pt'
        lang = 'en'
        spacy = 'en_core_web_sm'

        wsd = disambiguate.Disambiguator(spacy_checkpoint, lang=lang)  # , batch_size=5, save_wsd_details=False).eval()
        # wsd = wsd.to('cuda')
        nlp = load(spacy or lang, disable=['parser', 'ner'])
        wsd.enable(nlp, "wsd")

        print('== finding synonyms ==')

        cached_synonyms = {
            sentence.input: {
            # ^^ context ^^
                w.text.lower(): ([syn.replace('_', ' ') for syn in w._.synset._lemma_names if syn != w.lemma_] if w._.offset else [])
                # ^^  word  ^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ synonyms ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  if synset found
                for w in nlp(sentence.input)
            }
            for sentence in tqdm(sentences, desc='Generating Synonyms')
        }

        if conjugate_synonyms:
            def conjugate_synonym(word, synonym):
                """returns `synonym` conjugated using the tense of `word` if possible; else returns `synonym`"""
                try:
                    tense = tenses(word)[0]
                    synonym_parts = synonym.split(' ')  # when synonym is composed of several words, conjugate first one
                    synonym_parts[0] = conjugate(synonym_parts[0], *tense)
                    return ' '.join(synonym_parts)
                except IndexError:  # no tense -> not a verb
                    return synonym
                except Exception as e:
                    print(f'skipping synonym conjugation: for {word}->{synonym} due to error: {e}')
                    return synonym

            cached_synonyms = {
                context: {word: [conjugate_synonym(word, syn)
                                 for syn in syns]
                          for word, syns in D.items()}
                for context, D in cached_synonyms.items()}

        print(f'== saving synonyms cache to {synonyms_out_file_name} ==')

        with open(synonyms_out_file_name, 'w', encoding="utf-8") as f:
            json.dump(cached_synonyms, f, ensure_ascii=False)

    elif synonyms_method == "wordnet lesk":
        from nltk.wsd import lesk
    else:
        raise NotImplementedError(synonyms_method)

    def synonymed(keywords, sentence, n_outputs, method):
        # random.shuffle(i_words)
        augmentations = [keywords.copy() for i in range(n_outputs)]
        for i_word in range(len(keywords)):
            if method == 'ewiser' or 'cached' in method:
                synonyms = cached_synonyms[sentence][keywords[i_word].lower()]
            elif method == "wordnet lesk":
                synset = lesk(s.input, keywords[i_word])
                synonyms = [] if synset is None else\
                    [lemma.name() for lemma in synset.lemmas() if lemma.name() != keywords[i_word]]
            else:
                raise NotImplementedError(method)
            random.shuffle(synonyms)

            i_syn = 0
            for i_aug in random.sample(range(n_outputs), k=n_outputs):
                if synonyms and random.random() < p_synonyms:
                    synonym = synonyms[i_syn % len(synonyms)]
                    augmentations[i_aug][i_word] = synonym
                    i_syn += 1
        return augmentations

    for s in tqdm(sentences, desc='Adding Synonyms'):
        s.keywords_synonyms = synonymed(s.keywords, s.input, n_outputs, method=synonyms_method)


    print('=== shuffling ===')

    def shuffled(keywords, p_shuffles):
        if p_shuffles == 0:
            return keywords.copy()
        else:
            indices = np.arange(len(keywords))
            shuffle_me = np.random.rand(len(keywords)) < p_shuffles
            indices[shuffle_me] = np.random.permutation(indices[shuffle_me])
            return [keywords[i] for i in indices]

    for s in tqdm(sentences, desc='Shuffling'):
        s.keywords_shuffled = [shuffled(augmentation, p_shuffles) for augmentation in s.keywords_synonyms]


    if keep_punctuation:
        print('=== keeping punctuation ===')

        for s in tqdm(sentences, desc='Keeping punctuation'):
            if s.input.endswith(('.', '?', '!')):
                for augmentation in s.keywords_shuffled:
                    augmentation.append(s.input[-1])

    with open(agmentations_out_file_name, 'w', encoding="utf-8") as f:
        json.dump([dataclass_as_dict(s) for s in sentences], f, ensure_ascii=False)

else:
    with open(agmentations_out_file_name, encoding="utf-8") as f:
        sentences = [Sentence(**d) for d in json.load(f)]

if not baseline:
    assert all([len(s.keywords_synonyms) == n_outputs for s in sentences])
    assert all([len(s.keywords_shuffled) == n_outputs for s in sentences])


def run_pointer(pointer_model, out_file_txt, out_file_jsonl):
    print('=== make a sentence with POINTER ===')
    print('model is', pointer_model)

    with open(pointer_in_file_name, 'w', encoding="utf-8") as f:
        for s in sentences:
            for augmentation in s.keywords_shuffled:
                f.write('|'.join(augmentation)+'\n')
    #exit()

    command = ' '.join([
        pointer_env, f'{pointer_path}/inference_modified.py',
        '--bert_model', pointer_model,
        '--type', sample,
        '--no_ins_at_end', no_ins_at_end,
        '--keyfile', pointer_in_file_name,
        '--sep', '"|"',
        '--output_file', pointer_out_file_name,
    ])
    #sys.path.insert(0, pointer_path)
    os.environ['PYTHONPATH'] = f"{pointer_path}:{os.environ['PYTHONPATH']}"
    run_and_exit_if_fail(command)

if do_pointer:
    pass#run_pointer(pointer_model, out_file_txt, out_file_jsonl)
if do_diversity_eval and not baseline:
    with open(pointer_out_file_name, encoding="utf-8") as f:
        for s in sentences:
            s.outputs = [f.readline().rstrip('\n').rstrip('\r') for augmentation in s.keywords_shuffled]


if verbose:
    print('=== debugging results ===')

    def vprint(*args, **kwargs):
        print(*args, file=f, **kwargs)

    with _open(verbose_file, 'w') as f:
        for s in sentences:
            vprint('-')
            vprint('input:        ', s.input)
            vprint('keywords:     ', ' '.join(s.keywords))
            if do_pointer:
                for synonym, shuffle, output in zip(s.keywords_synonyms, s.keywords_shuffled, s.outputs):
                    vprint(' | synonym:   ', ' '.join(synonym))
                    vprint(' | | shuffle: ', ' '.join(shuffle))
                    if not print_outputs_after:
                        vprint(' | | | output:', output)
                if print_outputs_after:
                    vprint('outputs:', *s.outputs, sep='\n')


if (do_pointer and do_out_to_file) or baseline:
    print(f'=== writing results to {out_file_txt} and {out_file_jsonl} ===')

    with _open(out_file_txt, 'w') as f, _open(out_file_jsonl, 'w') as fjsonl:
        for s in tqdm(sentences, desc='Saving'):
            for output in s.outputs:
                print(output, file=f)
            json.dump({'src_text': s.input, 'tgt_texts': s.outputs}, fjsonl, ensure_ascii=False)
            print('', file=fjsonl)  # '\r\n'


if do_diversity_eval:
    print("=== showing paraphrase examples (5 randoms) ===")
    for s in random.sample(sentences, min(5, len(sentences))):
        print(s.input)
        for output in s.outputs:
            print(' |', output)

    print("=== showing paraphrase examples (5 first) ===")
    for s in sentences[:5]:
        print(s.input)
        for output in s.outputs:
            print(' |', output)
    
    print("=== computing paraphrase metrics ===")
    out_file_metrics = out_file_jsonl.rstrip('.jsonl') + '-metrics' + '.jsonl'
    command = f"{protaugment_env} -u " \
        f"evaluate_paraphrase_diversity.py " \
        f"--in_file {out_file_jsonl} --out_file {out_file_metrics}"  # should use os.path.abspath() ?
    os.environ['PYTHONPATH'] = f"{protaugment_path}:{os.environ['PYTHONPATH']}"
    run_and_exit_if_fail(command)

def run_protaugment(cv, c, k, dataset, regime):
    print('=== running protaugment ===')

    if 'protaugment' not in os.listdir(log_path):  # one dir per xp
        proto_log_path = f'{log_path}/protaugment'
    else:                                          # one shared dir
        proto_log_path = f'{log_path}/protaugment/{regime}/{dataset}/{cv:02d}/{c}C_{k}K/seed{protaugment_seed}' \
                         f'/pointer-paraphrase_{desc_paraphrase}_{desc_protaugment}'
    data_params = [
        "--data-path",                   f"{in_path}/full.jsonl",
        "--train-labels-path",           f"{in_path}/few_shot/{cv:02d}/labels.train.txt",
        "--valid-labels-path",           f"{in_path}/few_shot/{cv:02d}/labels.valid.txt",
        "--test-labels-path",            f"{in_path}/few_shot/{cv:02d}/labels.test.txt",
        "--unlabeled-path",              f"{in_path}/raw.txt",
    ] + (
        ["--train-path",                 f"{in_path}/few_shot/{cv:02d}/train.10samples.jsonl",
         "--unlabeled-path",             f"{in_path}/raw.txt"]
        if regime == '10samp' else []
    )
    pointer_paraphrase_params = [
        "--n-unlabeled",                 5,
        "--augmentation-data-path",      out_file_jsonl,
    ]
    model_params = [
        "--metric",                      "euclidean",
        "--supervised-loss-share-power", 1,
        "--model-name-or-path",          model_to_finetune,
    ]
    few_shot_params = [
        "--n-support", k,
        "--n-query", 5,
        "--n-classes", c,
    ]
    training_params = [
        "--evaluate-every",              100,
        "--n-test-episodes",             600,
        "--max-iter",                    10000,
        "--early-stop",                  8,
        "--log-every",                   10,
        "--seed",                        protaugment_seed,
    ]

    protaugment_main = f"{protaugment_path}/models/proto/protaugment.py"
    protaugment_custom_main = protaugment_main.rstrip('.py')+'_modified.py'

    command = ' '.join(
        [str(e) for e in
            [protaugment_env, '-u', protaugment_custom_main]
            + data_params
            + few_shot_params
            + training_params
            + pointer_paraphrase_params
            + model_params
            + ["--output-path", proto_log_path]
         ]
    )

    try:
        rmtree(proto_log_path)
        print(f'!!! Protaugment output file already exists. removing {proto_log_path} !!!')
    except FileNotFoundError:
        pass


    # inject a few lines to prevent tensorflow from eating all memory, leaving nothing for pytorh:
    if not os.path.isfile(protaugment_custom_main):
        os.system(f"cat limit_tensorflow_gpu_usage.py {protaugment_main} >> {protaugment_custom_main}")
        assert os.path.isfile(protaugment_custom_main)

    #sys.path.insert(0, protaugment_path)
    os.environ['PYTHONPATH'] = f"{protaugment_path}:{os.environ['PYTHONPATH']}"
    run_and_exit_if_fail(command)


    print('=== printing protaugment results as json ===')

    metrics_path = proto_log_path + '/metrics.json'
    with open(metrics_path, "r") as _f:
        metrics = json.load(_f)

    best_valid_acc = 0.0
    test_acc = 0.0
    i_eval_episode = 0
    assert len(metrics["valid"]) == len(metrics["test"])
    for i, (valid_episode, test_episode) in enumerate(zip(metrics["valid"], metrics["test"])):
        for valid_metric_dict in valid_episode["metrics"]:
            if valid_metric_dict["tag"].startswith("acc"):
                valid_acc = valid_metric_dict["value"]
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    i_eval_episode = i
                    for test_metric_dict in test_episode["metrics"]:
                        if test_metric_dict["tag"].startswith("acc"):
                            test_acc = test_metric_dict["value"]
                        break
                    else:
                        raise ValueError
                break
        else:
            raise ValueError

    results = {
        'dataset': {
            'name': dataset,
            'sentences': len(sentences),
            'paraphrases': n_outputs,
        },
        'keyword extraction': {
            'n keywords': n_keywords,
            'method': 'POINTER (yake+tf idf)' + ', then longest' if pick_longest_until_n_keywords else '',
            "keep '.?!'": keep_punctuation,
        },
        'synonyms': {
            'proba': p_synonyms,
            'method': synonyms_method,
        },
        'shuffles': {
            'proba': p_shuffles,
        },
        'POINTER': {
            'sampling': sample,
            'checkpoint': pointer_model,
            "sticky'.?!'": no_ins_at_end,
        },
        'ProtAugment': {
            'LM': model_to_finetune,
            'regime': regime,
            'run (cv)': cv,
            'C': c,
            'K': k,
            'seed': protaugment_seed,
        },
        'summary': {
            '': '',
        },
        'results': {
            'best episode': i_eval_episode,
            'valid acc': best_valid_acc,
            'test acc': test_acc,
        },
        'notes': {
            '': '',
        },
    }

    print(json.dumps(results, indent=4, ensure_ascii=False))


    print('=== printing protaugment results as csv ===')

    column_headers = [column for category in results for column in results[category]]
    column_values  = [results[category][column] for category in results for column in results[category]]
    print(*column_headers, sep=csv_sep)
    print(*column_values,  sep=csv_sep)

if do_protaugment:
    for regime, cv, c, k, dataset in product(regimes, CV, C, K, [dataset]):
        run_protaugment(cv, c, k, dataset, regime)


print('=== done! ===')
