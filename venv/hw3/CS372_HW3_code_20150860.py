import nltk
from nltk.corpus import brown, cmudict, wordnet
from collections import defaultdict
from tabulate import tabulate

tagged_corpus = brown.tagged_words()[:100]
tagged_sents = brown.tagged_sents()

pron_dict = cmudict.dict()


def count_same_spelling(sent):
    sent = list(map(lambda w: w[0].lower(), sent))
    wordset = set(sent)
    homos = dict(map(lambda w: (w, sent.count(w)), wordset))
    return dict(filter(lambda h: h[1] > 1, homos.items()))


def filter_single_pronunciation(words):
    prons = list(map(lambda h: (h[0], h[1], safe_prons(h[0])), words.items()))
    prons = list(map(lambda h: (h[0], h[1], normalize_stress(h[2])), prons))
    prons = list(map(lambda h: (h[0], h[1], len(h[2]), h[2]), prons))
    return list(filter(lambda h: h[2] > 1, prons))


def normalize_stress(prons):
    new_prons = set()
    for pron in prons:
        stress = list(map(lambda p: int(p[-1]) if p[-1].isdigit() else None, pron))
        mn = min(list(filter(lambda n: n is not None, stress)))
        stress = list(map(lambda s: s - mn if s is not None else None, stress))
        new_pron = tuple(map(lambda p: p[1][:-1] + str(stress[p[0]]) if stress[p[0]] is not None else p[1], enumerate(pron)))
        new_prons.add(new_pron)
    return new_prons


def filter_single_meaning(words):
    meanings = list(map(lambda w: (w[0], w[1], w[2], w[3], safe_meanings(w[0])), words))
    return list(filter(lambda w: len(w[4]) > 1, meanings))


def get_heteronyms(sent):
    print(' '.join(list(map(lambda w: w[0], sent))))
    spellings = count_same_spelling(sent)
    prons = filter_single_pronunciation(spellings)
    meanings = filter_single_meaning(prons)
    return meanings


def safe_prons(w):
    try:
        return pron_dict[w]
    except KeyError:
        return []


def safe_meanings(w):
    return list(filter(lambda s: w in s.lemma_names(), wordnet.synsets(w)))


heteronym_sents = list(map(lambda s: [s, get_heteronyms(s)], tagged_sents))
heteronym_sents = list(filter(lambda s: len(s[1]) > 0, heteronym_sents))
print(tabulate(heteronym_sents))
print(len(heteronym_sents), len(tagged_sents))
