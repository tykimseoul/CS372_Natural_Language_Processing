import nltk
from nltk.corpus import brown, cmudict, wordnet
from collections import defaultdict
from tabulate import tabulate
from itertools import combinations
from bs4 import BeautifulSoup
import requests
import re

sample_sents = [
    'A chair was so close to the door we couldn’t close it',
    'I wound a bandage around my wound',
    'When he wrecked his moped he moped all day',
    'Don’t just give the gift present the present',
    'The Polish man decided to polish his table',
    'She shed a tear because she had a tear in her shirt',
    'Farmers reap what they sow to feed it to the sow',
    'How much produce does the farm produce',
    'More people desert in the desert than in the mountains',
    'The researcher wanted to subject the subject to a psychology test'
]

tagged_sents = list(map(lambda s: (list(map(lambda w: (w, ''), s.split(' ')))), sample_sents))
# tagged_sents = brown.tagged_sents()[:100]
dictionary_url = 'https://www.dictionary.com/browse/'
pos_exclusions = ['\.', '\,', "''", ':', '--', '``', '\)', '\(']
pos_exclusions = "(" + ")|(".join(pos_exclusions) + ")"
word_exclusions = ['\w[0-9]+\w', '\w+-\w+']
word_exclusions = "(" + ")|(".join(word_exclusions) + ")"

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
        stress = stress_pattern(pron)
        mn = min(list(filter(lambda n: n is not None, stress)))
        stress = list(map(lambda s: s - mn if s is not None else None, stress))
        new_pron = tuple(map(lambda p: p[1][:-1] + str(stress[p[0]]) if stress[p[0]] is not None else p[1], enumerate(pron)))
        new_prons.add(new_pron)
    return new_prons


def stress_pattern(pron):
    return tuple(map(lambda p: int(p[-1]) if p[-1].isdigit() else None, pron))


def consonant_pattern(pron):
    return tuple(map(lambda p: p if not any(x in 'AEIOUY' for x in p) else None, pron))


def combine_by_condition(prons, condition):
    hashmap = dict(map(lambda p: (p, condition(p)), prons))
    unique_values = list(dict.fromkeys(hashmap.values()))
    indices = list(map(lambda v: list(hashmap.values()).index(v), unique_values))
    unique_keys = list(map(lambda v: list(hashmap.keys())[v], indices))
    return unique_keys


def filter_close_pronunciations(prons):
    prons = list(prons)
    new_prons = set()
    new_prons.update(combine_by_condition(prons, stress_pattern))
    new_prons.update(combine_by_condition(prons, consonant_pattern))
    return new_prons


def filter_single_meaning(words):
    meanings = list(map(lambda w: (w[0], w[1], w[2], w[3], safe_meanings(w[0])), words))
    return list(filter(lambda w: len(w[4]) > 1, meanings))


def get_heteronyms(sent):
    print(sent)
    sent = list(filter(lambda w: not re.match(word_exclusions, w[0]) and not re.match(pos_exclusions, w[1]), sent))
    heteronyms = list(map(lambda w: (w, crawl_pronunciation(w[0])), sent))
    heteronyms = list(filter(lambda w: w[1] is not None and len(w[1]) > 1, heteronyms))
    print('<HETERONYMS>', tabulate(heteronyms))
    return heteronyms


def safe_prons(w):
    try:
        return pron_dict[w]
    except KeyError:
        return []


def safe_meanings(w):
    return list(filter(lambda s: w in s.lemma_names(), wordnet.synsets(w)))


def crawl_pronunciation(word):
    word = word.lower()
    html = requests.get(dictionary_url + word)
    soup = BeautifulSoup(html.text, "html.parser")
    entries = soup.findAll(class_='entry-headword')
    if len(entries) == 0:
        print('========= invalid search:', word)
        return None
    definition_headers = soup.find_all('h2', attrs={'id': 'luna-section'})
    valid_definition_count = len(definition_headers) + 1 if definition_headers is not None else 1
    definitions = soup.findAll(class_='css-1urpfgu e16867sm0')
    definitions = list(filter(lambda d: get_entry_word(d) is not None, definitions))
    definitions = list(filter(lambda d: get_entry_word(d).text == word, definitions[:valid_definition_count]))
    ipas = []
    examples = []
    for definition in definitions:
        ipa = definition.find('span', attrs={'class': 'pron-ipa-content'})
        ipas.append(ipa)
        contents = definition.find_all('section', attrs={'class': 'css-pnw38j e1hk9ate0'})
        example = list(map(lambda c: extract_examples(c), contents))
        examples.append(example)
    prons = zip(ipas, examples)
    prons = list(filter(lambda i: i[0] is not None and i[1] is not None, prons))
    prons = list(map(lambda i: (simplify_ipa(i[0].text), i[1]), prons))
    if len(prons) > 1:
        collapsed = defaultdict(list)
        for pron in prons:
            collapsed[pron[0]].extend(pron[1])
        collapsed = list(map(lambda c: (c[0], c[1]), collapsed.items()))
        print(word, tabulate(collapsed))
        return collapsed
    else:
        print(word, tabulate(prons))
        return prons


def simplify_ipa(ipa):
    ipas = ipa.split(';')
    ipas = list(map(lambda i: i.replace('/', '').strip(), ipas))
    pos = ['verb', 'adjective', 'noun']
    if any(list(map(lambda i: any(s in i.lower() for s in pos), ipas))):
        pos_indices = list(map(lambda i: max(list(map(lambda p: i.lower().rfind(p) + len(p) if i.lower().rfind(p) is not -1 else -1, pos))), ipas))
        ipa_indices = list(map(lambda i: i[1].find(' ', pos_indices[i[0]] + 1), enumerate(ipas)))
        new_ipas = list(map(lambda i: i[1][:ipa_indices[i[0]]] if ipa_indices[i[0]] is not -1 else i[1], enumerate(ipas)))
        prons = defaultdict(list)
        for i in new_ipas:
            prons[i.split(' ')[-1]].extend(re.split('[,\s]', i)[:-1])
        new_prons = []
        for p in prons.items():
            for pos in p[1]:
                if len(pos) > 0:
                    new_prons.append((pos, p[0]))
        return tuple(new_prons)
    else:
        new_ipas = list(filter(lambda i: not any(s in i.lower() for s in ['stress', 'older', 'before']), ipas))
        if len(new_ipas) == 0:
            new_ipas = list(filter(lambda i: re.match('^.*unstressed.*$', i.lower()), ipas))[:1]
            if len(new_ipas) == 0:
                new_ipas = list(filter(lambda i: re.match('^.*consonant.*$', i.lower()), ipas))[:1]
        prons = tuple(map(lambda i: ('any', i), new_ipas))
        return prons


def get_entry_word(definition):
    return definition.find(['span', 'h1'], attrs={'class': 'css-1jzk4d9 e1rg2mtf8'})


def extract_examples(content):
    pos = content.find('span', attrs={'class': 'luna-pos'})
    if pos is not None:
        examples = content.find_all('span', attrs={'class': 'luna-example'})
        examples = list(map(lambda e: e.text, examples))
        return pos.text, examples
    else:
        return None, None


# different stress pattern or different consonant are different pronunciations, (different vowel but same consonant and same stress) is considered similar
'''
    considered different pronunciations:
        1. different stress pattern
        OR
        2. different consonants
'''
'''
    goal words:
        1. research
        2. juvenile
        3. replace
'''
# concatenate y pron with the next because that does not infer heteronym; heteronyms are not distinguished by addition of y
# look at the larger result and combine exceptions such as center, government, county etc
# pronunciations such as N+T->N, H+W->W; this is more of an accent difference than a pronunciation
# only one pronunciation should have it, N+T is followed by a vowel
# check consonant after changing and combine if same

heteronym_sents = list(map(lambda s: (s, get_heteronyms(s)), tagged_sents))
heteronym_sents = list(filter(lambda s: len(s[1]) > 0, heteronym_sents))
print(len(heteronym_sents), len(tagged_sents))
