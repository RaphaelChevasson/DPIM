from pattern.en import tenses, conjugate
assert conjugate('test'), "If this throws a StopIteration, please patch your pattern.en dependency: https://github.com/clips/pattern/issues/308#issuecomment-826404749"

word = 'broken'
synonym = 'go'

print( conjugate(synonym, *tenses(word)[0]) )


def conjugate_synonym(word, synonym):
    """returns `synonym` conjugated using the tense of `word` if possible; else returns `synonym`"""
    try:
        tense = tenses(word)[0]
        synonym_parts = synonym.split(' ')  # synonym of several words -> conjugate first word
        synonym_parts[0] = conjugate(synonym_parts[0], *tense)
        return ' '.join(synonym_parts)
    except IndexError:  # no tense -> not a verb
        return synonym
    except Exception as e:
        print(f'skipping synonym conjugation: for {word}->{synonym} due to error: {e}')
        return synonym

print( conjugate_synonym('broken', 'go bad') )
