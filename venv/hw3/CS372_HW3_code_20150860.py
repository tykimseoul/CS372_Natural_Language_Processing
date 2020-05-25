import nltk
from nltk.corpus import brown
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from tabulate import tabulate
from operator import itemgetter
from bs4 import BeautifulSoup
import requests
import re
import pprint

sample_sents = [
    'Medical intern and extern are employed and are good'
    # 'I wound a bandage around my wound',
    # 'How much produce does the farm produce',
    # 'She shed a tear because she had a tear in her shirt',
    # 'A wind is in the wind'
    # 'A chair was so close to the door we couldn ’t close it',
    # 'Farmers reap what they sow to feed it to the sow',
    # 'The Polish man decided to polish his table',
    # 'When he wrecked his moped he moped all day',
    # 'Don’t just give the gift present the present',
    # 'More people desert in the desert than in the mountains',
    # 'The researcher wanted to subject the subject to a psychology test'
]

pp = pprint.PrettyPrinter(indent=4)
stemmer = SnowballStemmer("english")

# mappings to simplify pos tags
pos_mapping = {'NN[A-Z]*': 'noun', 'JJ[A-Z]*': 'adjective', 'VB[A-Z]*': 'verb', 'RB[A-Z]*': 'adverb'}
simple_pos = {'NN[A-Z]*': 'NN', 'JJ[A-Z]*': 'JJ', 'VB[A-Z]*': 'VB', 'RB[A-Z]*': 'RB', 'DT': 'DT'}

tagged_sents = list(map(lambda s: nltk.pos_tag(nltk.word_tokenize(s)), sample_sents))
# tagged_sents = brown.tagged_sents()[:1]

dictionary_url = 'https://www.dictionary.com/browse/'

# regex expressions to ignore pos and word forms
pos_exclusions = ['\.', '\,', "''", ':', '--', '``', '\)', '\(']
pos_exclusions = "(" + ")|(".join(pos_exclusions) + ")"
word_exclusions = ['\w[0-9]+\w', '\w+-\w+']
word_exclusions = "(" + ")|(".join(word_exclusions) + ")"


# given a pos-tagged sentence, return list of words that can be used as heteronyms
def get_heteronyms(sent):
    print(sent)
    # filter out words in the sentence that should be ignored
    sent = list(filter(lambda w: not re.match(word_exclusions, w[0]) and not re.match(pos_exclusions, w[1]), sent))
    # crawl the pronunciation of each word
    heteronyms = list(map(lambda w: (w, crawl_pronunciation(w[0])), sent))
    # filter out words that have only single pronunciation
    heteronyms = list(filter(lambda w: w[1] is not None and count_pronunciations(w[1]) > 1, heteronyms))
    print('<HETERONYMS>', tabulate(heteronyms))
    return heteronyms


# given pronunciations, count the number of them
def count_pronunciations(prons):
    return sum(list(map(lambda p: len(p[0]), prons)))


# given a sentence and the heteronyms within, return the pronunciation-annotated string
def pronounce(sent, heteronyms):
    # the resulting sentence
    sentence = []
    # extract word part of heteronyms
    heteronym_words = list(map(lambda h: h[0], heteronyms))
    # form trigrams of the sentence to compare the surrounding pos tags
    trigrams = form_trigrams(sent)
    # for each word in the sentence
    for idx, word in enumerate(sent):
        if word in heteronym_words:
            # get pos tag of the heteronym
            pos = map_pos(word[1], pos_mapping)
            # get definition entries of the word
            heteronym_meaning = list(filter(lambda h: h[0] == word, heteronyms))
            # filter out definitions that does not include pos of the word
            relevant_definitions = list(filter(lambda d: pos in extract_pos(d), heteronym_meaning[0][1]))
            # print('relevant def for', word)
            # pp.pprint(relevant_definitions)
            # if no definition is used as the pos of the word, do not annotate it
            if len(relevant_definitions) == 0:
                sentence.append(word[0])
                continue
            # get a trigram for this word, e.g. words before and after
            trigram = trigrams[idx]
            # list of possible pronunciations
            candidates = []
            # for each relevant definition
            for definition in relevant_definitions:
                # mapping of pos and pronunciation(s)
                prons = dict(map(lambda d: (d[0], d[1]), definition[0]))
                # example sentences of current definition
                examples = dict(map(lambda e: (e[0], e[1]), definition[1]))
                # filter out example sentences that are not used as this word's pos
                examples = dict(filter(lambda e: pos in e[0], examples.items()))
                if pos in prons.keys():
                    # if pronunciation mapping includes this word's pos, add to possible pronunciations
                    candidates.append((prons[pos], examples))
                elif 'any' in prons.keys():
                    # if all pos is pronounced identically, add to possible pronunciations
                    candidates.append((prons['any'], examples))
                else:
                    # all other cases, do not annotate
                    sentence.append(word[0])
                    break
            # ensure there is at least one pronunciation
            assert len(candidates) > 0
            if len(candidates) == 1:
                # if there is only a single possible pronunciation, take it and continue to next word in the sentence
                sentence.append(str(word) + '[' + candidates[0][0] + ']')
                continue
            # make a map of pos and corresponding example sentences
            flat_examples = list(map(lambda c: (c[0], c[1].values()), candidates))
            flat_examples = list(map(lambda c: (c[0], [item for sublist in c[1] for item in sublist]), flat_examples))
            print('flat', flat_examples)
            if not any(list(map(lambda e: len(e[1]) > 0, flat_examples))) or trigram is None:
                # if there are no example sentences available or if the trigram could not be formed, take the first possible pronunciation
                sentence.append(str(word) + '[' + candidates[0][0] + ']')
                continue
            # calculate pos similarity of trigrams to find the most probable pronunciation
            candidate_similarity = list(map(lambda e: (e[0], measure_similarity(trigram, e[1])), flat_examples))
            candidate_similarity = sorted(candidate_similarity, key=itemgetter(1), reverse=True)
            print(tabulate(candidate_similarity))
            # take the pronunciation with highest similarity
            sentence.append(str(word) + '[' + candidate_similarity[0][0] + ']')
        else:
            # if the word is not a heteronym, do not annotate it
            sentence.append(word[0])
    print(' '.join(sentence))
    # return the joined string
    return ' '.join(sentence)


# given a sentence, return the list of trigram in which first and last ones are None
def form_trigrams(sent):
    trigrams = list(nltk.trigrams(sent))
    trigrams.insert(0, None)
    trigrams.append(None)
    return trigrams


# given a trigram of target heteronym word and the list of example sentences, score the similarity of the uses
def measure_similarity(target, examples):
    # tokenize example sentences
    examples = list(map(lambda e: nltk.word_tokenize(e), examples))
    # pos tag the sentences
    examples = list(map(lambda e: nltk.pos_tag(e), examples))
    # form trigrams of each sentence
    examples = list(map(lambda e: list(nltk.trigrams(e)), examples))
    # get trigrams that contain the same center word as the target
    examples = list(map(lambda e: list(filter(lambda t: stemmize(t[1]) == stemmize(target[1]), e)), examples))
    # flatten the list of trigrams
    examples = [item for sublist in examples for item in sublist]
    print('ex', examples)
    if len(examples) == 0:
        # if there are no trigrams that meet the criteria return 0 similarity
        return 0
    # extract pos of target trigram
    target = list(map(lambda t: t[1], target))
    # count number of matches with first and last pos of the target trigram in the list of example trigrams
    first_matches = len(list(filter(lambda e: map_pos(e[0][1], simple_pos) == map_pos(target[0], simple_pos), examples)))
    second_matches = len(list(filter(lambda e: map_pos(e[2][1], simple_pos) == map_pos(target[2], simple_pos), examples)))
    # return calculated similarity
    return (first_matches + second_matches) / (len(examples) * 2)


# stemmize a word-pos tuple to compare for similarity
def stemmize(word):
    return stemmer.stem(word[0]), map_pos(word[1], simple_pos)


# given a definition entry return the list of pos within
def extract_pos(definitions):
    poses = list(map(lambda d: d[0], definitions[1]))
    poses = list(map(lambda p: re.sub('\([^)]*\)', '', p), poses))
    poses = list(map(lambda p: re.sub('[^\w\s]', '', p).strip(), poses))
    return poses


# map pos (NN, VB, etc) to (noun, verb, etc)
def map_pos(pos, mapping):
    mapped = list(filter(lambda p: re.match(p[0], pos), mapping.items()))
    if len(mapped) == 0:
        print('======== invalid pos:', pos)
        return None
    return mapped[0][1]


# crawl pronunciation of the provided word
def crawl_pronunciation(word):
    word = word.lower()
    html = requests.get(dictionary_url + word)
    soup = BeautifulSoup(html.text, "html.parser")
    entries = soup.findAll(class_='entry-headword')
    if len(entries) == 0:
        print('========= invalid search:', word)
        return None
    # I want only American-English entries
    definition_headers = soup.find_all('h2', attrs={'id': 'luna-section'})
    valid_definition_count = len(definition_headers) + 1 if definition_headers is not None else 1
    definitions = soup.findAll(class_='css-1urpfgu e16867sm0')
    definitions = list(filter(lambda d: get_entry_word(d) is not None, definitions))
    # ignore results that are different from the provided word
    definitions = list(filter(lambda d: get_entry_word(d).text.lower() == word, definitions[:valid_definition_count]))
    ipas = []
    examples = []
    for definition in definitions:
        # get ipa string
        ipa = definition.find('span', attrs={'class': 'pron-ipa-content'})
        ipas.append(ipa)
        contents = definition.find_all('section', attrs={'class': 'css-pnw38j e1hk9ate0'})
        # get example sentences of each definition
        example = list(map(lambda c: extract_examples(c), contents))
        examples.append(example)
    # zip ipa to examples
    prons = zip(ipas, examples)
    prons = list(filter(lambda i: i[0] is not None and i[1] is not None, prons))
    # simplify ipa string of various forms
    prons = list(map(lambda i: (simplify_ipa(i[0].text), i[1]), prons))
    if len(prons) > 1:
        # if there are multiple pronunciations, collapse the list
        collapsed = defaultdict(list)
        for pron in prons:
            collapsed[pron[0]].extend(pron[1])
        collapsed = list(map(lambda c: (c[0], c[1]), collapsed.items()))
        print(word, tabulate(collapsed))
        return collapsed
    else:
        # otherwise, simply return it
        print(word, tabulate(prons))
        return prons


# the ipa string have various forms; preprocess it to a simpler version
def simplify_ipa(ipa):
    ipas = ipa.split(';')
    ipas = list(map(lambda i: i.replace('/', '').strip(), ipas))
    pos = ['verb', 'adjective', 'noun']
    if any(list(map(lambda i: any(s in i.lower() for s in pos), ipas))):
        # if ipa specifies pos for each pronunciation, reshape it into a dictionary
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
        # if ipa contains stress, history or position information, choose the unstressed and used before-a-consonant version
        new_ipas = list(filter(lambda i: not any(s in i.lower() for s in ['stress', 'older', 'before']), ipas))
        if len(new_ipas) == 0:
            new_ipas = list(filter(lambda i: re.match('^.*unstressed.*$', i.lower()), ipas))[:1]
            if len(new_ipas) == 0:
                new_ipas = list(filter(lambda i: re.match('^.*consonant.*$', i.lower()), ipas))[:1]
        new_ipas = list(filter(lambda i: not i.lower().startswith('for'), new_ipas))
        prons = tuple(map(lambda i: ('any', i), new_ipas))
        return prons


# crawl the name of the entry word using the css class
def get_entry_word(definition):
    return definition.find(['span', 'h1'], attrs={'class': 'css-1jzk4d9 e1rg2mtf8'})


# crawl the example sentences of entry using the css class
def extract_examples(content):
    pos = content.find('span', attrs={'class': 'luna-pos'})
    if pos is not None:
        examples = content.find_all('span', attrs={'class': 'luna-example'})
        examples = list(map(lambda e: e.text, examples))
        return pos.text, examples
    else:
        return None, None


# score each sentence in terms of heteronyms
def score(heteronyms):
    heteronym_words = list(map(lambda h: (h[0][0], map_pos(h[0][1], pos_mapping)), heteronyms))
    heteronym_words = list(filter(lambda w: w[1] is not None, heteronym_words))
    # number of heteronym occurrences (criteria #1)
    count = len(heteronym_words)
    heteronym_set = set(map(lambda w: w[0], heteronym_words))
    # number of distinct heteronyms (criteria #2)
    kinds = len(heteronym_set)
    # number of heteronyms with the same pos (criteria #3)
    pos_variation = len(set(heteronym_words))
    unit = 100
    # high score means high count, low kinds and low pos_variation
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

# find heteronyms of each sentence
heteronym_sents = list(map(lambda s: (s, get_heteronyms(s)), tagged_sents))
# filter out sentences without heteronyms
heteronym_sents = list(filter(lambda s: len(s[1]) > 0, heteronym_sents))
# annotate pronunciations of each sentence
heteronym_sents = list(map(lambda s: (s[0], s[1], pronounce(s[0], s[1])), heteronym_sents))
# score the sentences
heteronym_sents = list(map(lambda s: (s[2], score(s[1])), heteronym_sents))
# rank the sentences
heteronym_sents = sorted(heteronym_sents, key=itemgetter(1), reverse=True)
print(tabulate(heteronym_sents))
print(len(heteronym_sents), len(tagged_sents))

# save to file
f = open('CS372_HW3_output_20150860.csv', 'w')
for t in heteronym_sents[:30]:
    f.write("{}, Brown\n".format(t[0]))
