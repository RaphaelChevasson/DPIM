from ewiser.spacy import disambiguate
from spacy import load

spacy_checkpoint = '../ewiser/ckpt/ewiser.semcor+wngt.pt'
lang = 'en'
spacy = 'en_core_web_sm'


wsd = disambiguate.Disambiguator(spacy_checkpoint, lang=lang)  # , batch_size=5, save_wsd_details=False).eval()
# wsd = wsd.to('cuda')
nlp = load(spacy or lang, disable=['parser', 'ner'])
wsd.enable(nlp, "wsd")


last_sentence = None

def get_tokenizer():
    return lambda sentence: [str(word) for word in nlp.tokenizer(sentence)]

def get_synonyms(word, sentence, verbose=False):
    global last_sentence, doc, ewiser_data

    # process sentence with ewiser (cached)
    if last_sentence is None or sentence != last_sentence:
        doc = nlp(sentence)
        ewiser_data = {}
        for w in doc:
            ewiser_data[w.text.lower()] = w

    # return word synonyms
    w = ewiser_data[word.lower()]
    if w._.offset:  # we found a synset in wordnet
        # print(w.text, w.lemma_, w.pos_, w._.offset, w._.synset.definition())
        synonyms = [s for s in w._.synset._lemma_names if s != w.lemma_]
        if verbose and w.lemma_ != w.text and len(w._.synset._lemma_names) > 1:
            print(f"Warning: replacing non-lemmatized word `{word}` by lemmatized synonyms `{synonyms}` in sentnce `{sentence}`")
        return synonyms
    else:
        return []
