import nltk
from nltk.corpus import brown, cmudict, wordnet
from collections import defaultdict
from tabulate import tabulate
from itertools import combinations

tagged_sents = brown.tagged_sents()[:1000]

pron_dict = cmudict.dict()


def count_same_spelling(sent):
    sent = list(map(lambda w: w[0].lower(), sent))
    wordset = set(sent)
    homos = dict(map(lambda w: (w, sent.count(w)), wordset))
    return dict(filter(lambda h: h[1] > 1, homos.items()))


def filter_single_pronunciation(words):
    prons = list(map(lambda h: (h[0], h[1], safe_prons(h[0])), words.items()))
    prons = list(map(lambda h: (h[0], h[1], concatenate_y(h[2])), prons))
    prons = list(map(lambda h: (h[0], h[1], normalize_stress(h[2])), prons))
    prons = list(map(lambda h: (h[0], h[1], filter_close_pronunciations(h[2])), prons))
    prons = list(map(lambda h: (h[0], h[1], len(h[2]), h[2]), prons))
    return list(filter(lambda h: h[2] > 1, prons))


def concatenate_y(prons):
    new_prons = []
    for pron in prons:
        if len(pron) == 1:
            new_prons.append(pron)
            continue
        new_pron = list(map(lambda p: None if p == 'Y' else p, pron))
        new_pron = list(map(lambda p: p[1] if pron[p[0] - 1] != 'Y' else 'Y' + p[1], enumerate(new_pron)))
        new_pron = list(filter(lambda p: p is not None, new_pron))
        new_prons.append(new_pron)
    return new_prons


def normalize_stress(prons):
    new_prons = set()
    for pron in prons:
        stress = extract_stress(pron)
        mn = min(list(filter(lambda n: n is not None, stress)))
        stress = list(map(lambda s: s - mn if s is not None else None, stress))
        new_pron = tuple(map(lambda p: p[1][:-1] + str(stress[p[0]]) if stress[p[0]] is not None else p[1], enumerate(pron)))
        new_prons.add(new_pron)
    return new_prons


def extract_stress(pron):
    return list(map(lambda p: int(p[-1]) if p[-1].isdigit() else None, pron))


def combine_stress(pron1, pron2):
    if extract_stress(pron1) != extract_stress(pron2):
        return {pron1, pron2}
    else:
        return {pron1}


def extract_consonants(pron):
    return list(map(lambda p: p if not any(x in 'AEIOUY' for x in p) else None, pron))


def combine_consonants(pron1, pron2):
    if extract_consonants(pron1) != extract_consonants(pron2):
        return {pron1, pron2}
    else:
        return {pron1}


def filter_close_pronunciations(prons):
    new_prons = set()
    for pair in combinations(prons, 2):
        new_prons.update(combine_stress(pair[0], pair[1]))
        new_prons.update(combine_consonants(pair[0], pair[1]))
    return new_prons


def filter_single_meaning(words):
    meanings = list(map(lambda w: (w[0], w[1], w[2], w[3], safe_meanings(w[0])), words))
    return list(filter(lambda w: len(w[4]) > 1, meanings))


def get_heteronyms(sent):
    spellings = count_same_spelling(sent)
    prons = filter_single_pronunciation(spellings)
    meanings = filter_single_meaning(prons)
    if len(meanings) > 0:
        print(' '.join(list(map(lambda w: w[0], sent))))
        print(meanings)
    return meanings


def safe_prons(w):
    try:
        return pron_dict[w]
    except KeyError:
        return []


def safe_meanings(w):
    return list(filter(lambda s: w in s.lemma_names(), wordnet.synsets(w)))


# different stress pattern or different consonant, different vowel is considered similar
# concatenate y pron with the next because that does not infer heteronym; heteronyms are not distinguished by addition of y

heteronym_sents = list(map(lambda s: [s, get_heteronyms(s)], tagged_sents))
heteronym_sents = list(filter(lambda s: len(s[1]) > 0, heteronym_sents))
print(tabulate(heteronym_sents))
print(len(heteronym_sents), len(tagged_sents))
