import nltk
from collections import defaultdict
from tabulate import tabulate
from operator import itemgetter
from bs4 import BeautifulSoup
import requests
import re
import pprint

sample_sents = [
    'I wound a bandage around my wound',
    'How much produce does the farm produce',
    'She shed a tear because she had a tear in her shirt',
    'A chair was so close to the door we couldn ’t close it',
    'Farmers reap what they sow to feed it to the sow',
    'The Polish man decided to polish his table',
    'When he wrecked his moped he moped all day',
    'Don’t just give the gift present the present',
    'More people desert in the desert than in the mountains',
    'The researcher wanted to subject the subject to a psychology test'
]

pp = pprint.PrettyPrinter(indent=4)

pos_mapping = {'NN[A-Z]*': 'noun', 'JJ[A-Z]*': 'adjective', 'VB[A-Z]*': 'verb', 'RB[A-Z]*': 'adverb'}

tagged_sents = list(map(lambda s: nltk.pos_tag(nltk.word_tokenize(s)), sample_sents))
# tagged_sents = brown.tagged_sents()[:100]
dictionary_url = 'https://www.dictionary.com/browse/'
pos_exclusions = ['\.', '\,', "''", ':', '--', '``', '\)', '\(']
pos_exclusions = "(" + ")|(".join(pos_exclusions) + ")"
word_exclusions = ['\w[0-9]+\w', '\w+-\w+']
word_exclusions = "(" + ")|(".join(word_exclusions) + ")"


def get_heteronyms(sent):
    print(sent)
    sent = list(filter(lambda w: not re.match(word_exclusions, w[0]) and not re.match(pos_exclusions, w[1]), sent))
    heteronyms = list(map(lambda w: (w, crawl_pronunciation(w[0])), sent))
    heteronyms = list(filter(lambda w: w[1] is not None and count_pronunciations(w[1]) > 1, heteronyms))
    print('<HETERONYMS>', tabulate(heteronyms))
    return heteronyms


def count_pronunciations(prons):
    return sum(list(map(lambda p: len(p[0]), prons)))


def pronounce(sent, heteronyms):
    sentence = []
    for word in sent:
        heteronym_words = list(map(lambda h: h[0], heteronyms))
        if word in heteronym_words:
            pos = map_pos(word[1])
            heteronym_meaning = list(filter(lambda h: h[0] == word, heteronyms))
            relevant_definitions = list(filter(lambda d: pos in extract_pos(d), heteronym_meaning[0][1]))
            # print('relevant def for', word)
            # pp.pprint(relevant_definitions)
            if len(relevant_definitions) == 0:
                sentence.append(word[0])
                continue
            for definition in relevant_definitions:
                prons = dict(map(lambda d: (d[0], d[1]), definition[0]))
                if pos in prons.keys():
                    sentence.append(word[0] + '[' + prons[pos] + ']')
                    break
                elif 'any' in prons.keys():
                    sentence.append(word[0] + '[' + prons['any'] + ']')
                    break
                else:
                    sentence.append(word[0])
                    break
        else:
            sentence.append(word[0])
    print(' '.join(sentence))
    return ' '.join(sentence)


def extract_pos(definitions):
    poses = list(map(lambda d: d[0], definitions[1]))
    poses = list(map(lambda p: re.sub('\([^)]*\)', '', p), poses))
    poses = list(map(lambda p: re.sub('[^\w\s]', '', p).strip(), poses))
    return poses


def map_pos(pos):
    mapped = list(filter(lambda p: re.match(p[0], pos), pos_mapping.items()))
    if len(mapped) == 0:
        print('======== invalid pos:', pos)
        return None
    return mapped[0][1]


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
    definitions = list(filter(lambda d: get_entry_word(d).text.lower() == word, definitions[:valid_definition_count]))
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


def score(heteronyms):
    heteronym_words = list(map(lambda h: (h[0][0], map_pos(h[0][1])), heteronyms))
    heteronym_words = list(filter(lambda w: w[1] is not None, heteronym_words))
    count = len(heteronym_words)
    heteronym_set = set(map(lambda w: w[0], heteronym_words))
    kinds = len(heteronym_set)
    pos_variation = len(set(heteronym_words))
    unit = 100
    score = count * unit ** 2 + (unit - kinds) * unit + (unit - pos_variation)
    # print(count, kinds, pos_variation, score)
    return score


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
heteronym_sents = list(map(lambda s: (s[0], s[1], pronounce(s[0], s[1])), heteronym_sents))
heteronym_sents = list(map(lambda s: (s[2], score(s[1])), heteronym_sents))
heteronym_sents = sorted(heteronym_sents, key=itemgetter(1), reverse=True)
print(tabulate(heteronym_sents))
print(len(heteronym_sents), len(tagged_sents))
