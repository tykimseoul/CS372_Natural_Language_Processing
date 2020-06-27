import pandas as pd
import nltk
import re
from itertools import accumulate
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
import time
from multiprocessing import Pool, cpu_count, set_start_method
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 3000)
pd.set_option('display.max_rows', None)

exclusions = ['JJ.?', 'RB.?']
# join into a single regex
exclusions = "(" + ")|(".join(exclusions) + ")"

all_male_names = defaultdict(set)
all_female_names = defaultdict(set)


def word_tokenize_with_dash(sent):
    # if '-' in sent:
    #     sent = re.sub(r"-", " - ", sent)
    #     sent = re.sub(r"\s+", " ", sent)
    #     sent = re.sub(r"- -", "--", sent)
    sent = re.sub(r"\.,", ",", sent)
    return nltk.word_tokenize(sent)


def get_word_index(sent, entity, offset=0):
    entity_tokens = word_tokenize_with_dash(entity)
    word_offset = len(word_tokenize_with_dash(sent[:offset]))
    sent = sent[offset:]
    sent_tokens = word_tokenize_with_dash(sent)
    if len(entity_tokens) > 1:
        ngrams = list(nltk.ngrams(sent_tokens, len(entity_tokens)))
        if tuple(entity_tokens) in ngrams:
            idx = ngrams.index(tuple(entity_tokens))
            return [idx + i + word_offset for i in range(len(entity_tokens))]
        else:
            return [-1]
    else:
        if entity_tokens[0] in sent_tokens:
            return [list(sent_tokens).index(entity_tokens[0]) + word_offset]
        else:
            return [-1]


def get_sent_index(text, entity, offset=0):
    sents = nltk.sent_tokenize(text)
    lengths = list(map(len, sents))
    lengths_acc = [0] + list(accumulate(lengths))
    sent_index = lengths_acc.index(next(filter(lambda l: l >= offset, lengths_acc))) - 1
    offset -= (lengths_acc[sent_index] + sent_index)
    word_index = get_word_index(sents[sent_index], entity, offset)
    return sent_index, word_index


# remove adjectives and adverbs from the provided sentence
def simplify_sentence(sent):
    sent = nltk.pos_tag(word_tokenize_with_dash(sent))
    sent = ' '.join(list(map(lambda w: w[0] if not re.match(exclusions, w[1]) else '#', sent)))
    sent = re.sub(r'\s([,.?;\'])', r'\1', sent)
    return sent


# remove sentences that does not contain the pronoun, A or B
def reduce_text(text, relevant_indices):
    sents = nltk.sent_tokenize(text)
    relevant_sents = list(filter(lambda s: s[0] in relevant_indices, enumerate(sents)))
    relevant_sents = list(map(lambda s: s[1], relevant_sents))
    return ' '.join(relevant_sents)


def relevant_sentences(text, pronoun, a, b):
    indices = [pronoun, a, b]
    sents = nltk.sent_tokenize(text)
    lengths = list(map(len, sents))
    lengths_acc = [0] + list(accumulate(lengths))
    relevant_indices = set(map(lambda i: lengths_acc.index(next(filter(lambda l: l > i, lengths_acc))) - 1, indices))
    return relevant_indices


def get_simplified_sent_index(text, sent_index, relevant_indices):
    irrelevant_indices = set(range(len(nltk.sent_tokenize(text)))) - relevant_indices
    irrelevant_count_before_current = len(set(filter(lambda i: i < sent_index[0], irrelevant_indices)))
    return sent_index[0] - irrelevant_count_before_current, sent_index[1]


def label_entities(text, pronoun, a, b):
    pronoun = (pronoun[0], tuple(pronoun[1]))
    a = (a[0], tuple(a[1]))
    b = (b[0], tuple(b[1]))
    sents = nltk.sent_tokenize(text)
    sents = list(map(word_tokenize_with_dash, sents))
    labels = {pronoun: 'PRN', a: 'A', b: 'B'}
    for label in labels.keys():
        for idx in label[1]:
            sents[label[0]][idx] = (sents[label[0]][idx], labels[label])
    return sents


def read_tsv(filename):
    df = pd.read_csv(filename, sep='\t')
    return df


def find_indices(df):
    df['Pronoun-index'] = df.apply(lambda r: get_word_index(r['Text'], r['Pronoun'], r['Pronoun-offset']), axis=1)
    df['A-index'] = df.apply(lambda r: get_word_index(r['Text'], r['A'], r['A-offset']), axis=1)
    df['B-index'] = df.apply(lambda r: get_word_index(r['Text'], r['B'], r['B-offset']), axis=1)
    df['Pronoun-sent-index'] = df.apply(lambda r: get_sent_index(r['Text'], r['Pronoun'], r['Pronoun-offset']), axis=1)
    df['A-sent-index'] = df.apply(lambda r: get_sent_index(r['Text'], r['A'], r['A-offset']), axis=1)
    df['B-sent-index'] = df.apply(lambda r: get_sent_index(r['Text'], r['B'], r['B-offset']), axis=1)
    return df


def simplify(df):
    df['Relevant-sentences'] = df.apply(lambda r: relevant_sentences(r['Text'], r['Pronoun-offset'], r['A-offset'], r['B-offset']), axis=1)
    df['Text-simplified'] = df.apply(lambda r: reduce_text(r['Text'], r['Relevant-sentences']), axis=1)
    df['Pronoun-sent-index-simplified'] = df.apply(lambda r: get_simplified_sent_index(r['Text'], r['Pronoun-sent-index'], r['Relevant-sentences']), axis=1)
    df['A-sent-index-simplified'] = df.apply(lambda r: get_simplified_sent_index(r['Text'], r['A-sent-index'], r['Relevant-sentences']), axis=1)
    df['B-sent-index-simplified'] = df.apply(lambda r: get_simplified_sent_index(r['Text'], r['B-sent-index'], r['Relevant-sentences']), axis=1)
    df['Text-simplified'] = df.apply(lambda r: simplify_sentence(r['Text-simplified']), axis=1)
    df['Sent-simplified-lengths'] = df.apply(lambda r: [len(word_tokenize_with_dash(s)) for s in nltk.sent_tokenize(r['Text-simplified'])], axis=1)
    df['reduction'] = df.apply(lambda r: '{} -> {}'.format(len(nltk.sent_tokenize(r['Text'])), len(nltk.sent_tokenize(r['Text-simplified']))), axis=1)
    return df


def extract_candidates_snippet_context(sent, pronoun):
    sents = nltk.sent_tokenize(sent)
    # ignore those that appear in a different sentence after one that contains the pronoun
    # TODO: test this
    sents = sents[:pronoun[0] + 1]
    assert len(sents) > 0

    def single_sent_candidates(sent):
        print('raw', sent)
        untagged_sent = word_tokenize_with_dash(sent)
        sent = nltk.pos_tag(untagged_sent)
        cp = nltk.RegexpParser('NNP: {<NNP>+}')
        sent = cp.parse(sent)
        candidate = list(filter(lambda s: type(s) == nltk.tree.Tree and s.label() == 'NNP', sent))
        candidate = list(map(lambda t: t.flatten(), candidate))
        candidate = list(map(lambda t: t.leaves(), candidate))
        candidate = list(map(lambda l: ' '.join([p[0] for p in l]), candidate))
        genders = list(map(lambda e: check_wikipedia(e), candidate))
        positions = list(map(lambda e: list(nltk.ngrams(untagged_sent, len(word_tokenize_with_dash(e)))).index(tuple(word_tokenize_with_dash(e))), candidate))
        return list(zip(candidate, genders, positions))

    candidates = list(map(lambda s: single_sent_candidates(s), sents))
    result = []
    for s in enumerate(candidates):
        indexed = [(c[0], c[1], s[0], c[2]) for c in s[1]]
        result.extend(indexed)
    result = list(filter(lambda r: r[1][0] != 'NOT', result))
    return result


def check_wikipedia(entity):
    html = requests.get('https://en.wikipedia.org/wiki/{}'.format(entity))
    soup = BeautifulSoup(html.text, "html.parser")
    categories = soup.find(class_='mw-normal-catlinks')
    gender = 'UNK'
    person = 'NOT'
    if categories is None:
        return person, gender
    categories = categories.findAll('li')
    categories = list(map(lambda c: c.text, categories))
    if 'Disambiguation pages' in categories:
        headers = soup.findAll(class_='mw-headline')
        headers = list(map(lambda h: h.text, headers))
        is_name = 'People' in headers
        if is_name:
            person = 'NAME'
            gender = determine_gender(entity)
    else:
        is_real_person = any(list(map(lambda c: 'births' in c or 'people' in c or 'deaths' in c, categories)))
        if not is_real_person:
            gender = determine_gender(entity)
            if gender != 'UNK':
                person = 'NAME'
            else:
                person = 'NOT'
        else:
            person = 'REAL'
            gender = determine_gender(entity)
    result = (person, gender)
    return result


def determine_gender(name):
    name = word_tokenize_with_dash(name)
    # can determine first name only if more than one term is provided
    if len(name) > 1:
        is_male = gender_helper('https://en.wikipedia.org/w/index.php?title=Category:English_masculine_given_names&from={}'.format(name[0][0]), name[0], 0)
        is_female = gender_helper('https://en.wikipedia.org/w/index.php?title=Category:English_feminine_given_names&from={}'.format(name[0][0]), name[0], 1)
        if is_male != is_female:
            if is_male:
                return 'M'
            else:
                return 'F'
    return 'UNK'


def gender_helper(link, name, gender):
    all_names = {0: all_male_names, 1: all_female_names}
    first_letter = name[0].lower()
    if len(all_names[gender][first_letter]) == 0:
        html = requests.get(link)
        soup = BeautifulSoup(html.text, "html.parser")
        names = soup.findAll('li')
        names = list(map(lambda c: c.text, names))
        names = list(filter(lambda n: len(n) > 0, names))
        names = list(filter(lambda n: n[0].lower() == first_letter, names))
        contained = any(list(map(lambda c: name.lower() in word_tokenize_with_dash(c.lower()), names)))
        all_names[gender][first_letter].update(names)
    else:
        contained = any(list(map(lambda c: name.lower() in word_tokenize_with_dash(c.lower()), all_names[gender][first_letter])))
    return contained


def calculate_distance(lengths, pronoun, pronoun_index, candidates):
    def dist(pronoun_sent_idx, pronoun_word_idx, word_sent_idx, word_idx):
        if pronoun_sent_idx < word_sent_idx:
            return lengths[pronoun_sent_idx] - pronoun_word_idx + word_idx
        elif pronoun_sent_idx > word_sent_idx:
            return lengths[word_sent_idx] - word_idx + pronoun_word_idx
        else:
            return abs(pronoun_word_idx - word_idx)

    distances = list(map(lambda c: dist(pronoun_index[0], pronoun_index[1][0], c[2], c[3]), candidates))
    if pronoun.lower() == 'he' or pronoun.lower() == 'his':
        wrong_gender = list(map(lambda c: c[0] if c[1][1][1] == 'F' else -1, enumerate(candidates)))
    elif pronoun.lower() == 'she' or pronoun.lower() == 'her':
        wrong_gender = list(map(lambda c: c[0] if c[1][1][1] == 'M' else -1, enumerate(candidates)))
    else:
        wrong_gender = []
    distances = list(map(lambda d: 100000 if d[0] in wrong_gender else d[1], enumerate(distances)))
    return distances


def extract_snippet_context(df):
    df['candidates'] = df.apply(lambda r: extract_candidates_snippet_context(r['Text-simplified'], r['Pronoun-sent-index-simplified']), axis=1)
    return df


def choose_candidate(a, b, candidates, distances):
    a_guess = False
    b_guess = False
    if len(distances) == 0:
        return a_guess, b_guess
    min_idx = distances.index(min(distances))
    try:
        a_idx = list(map(lambda c: c[0], candidates)).index(a)
        a_guess = a_idx == min_idx
    except ValueError:
        a_idx = -1
    try:
        b_idx = list(map(lambda c: c[0], candidates)).index(b)
        b_guess = b_idx == min_idx
    except ValueError:
        b_idx = -1

    if a_idx == -1 and b_idx == -1:
        return a_guess, b_guess
    if a_guess == b_guess:
        a_dist = distances[a_idx]
        b_dist = distances[b_idx]
        if a_dist < b_dist:
            a_guess = True
            b_guess = False
        else:
            a_guess = False
            b_guess = True
    return a_guess, b_guess


def guess(df):
    df['distance'] = df.apply(lambda r: calculate_distance(r['Sent-simplified-lengths'], r['Pronoun'], r['Pronoun-sent-index-simplified'], r['candidates']), axis=1)
    df['guess'] = df.apply(lambda r: choose_candidate(r['A'], r['B'], r['candidates'], r['distance']), axis=1)
    df['A-guess'], df['B-guess'] = zip(*df.guess)
    df.drop('guess', axis=1, inplace=True)
    return df


def parallelize(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    df_split = list(filter(lambda d: len(d) > 0, df_split))
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == '__main__':
    set_start_method("spawn")
    # extract names from sentences and find nearest one to the pronoun. But exclude those that appear after and different sentence.
    # Test so that the last condition actually improves accuracy.
    # filter only names from the candidates by searching in wikipedia because named entities usually appear in wikipedia
    # in context aware cases, use the links already provided in the wikipedia page
    # if multiple terms, first one is the first name, if not, it can be either first name or last name
    # determine gender of name
    data = read_tsv('./GAP/gap-test.tsv')[:10]
    start = time.time()
    data = parallelize(data, find_indices, cpu_count())
    data = parallelize(data, simplify, cpu_count())
    data_snippet = parallelize(data, extract_snippet_context, min(4, cpu_count()))
    data_snippet = parallelize(data_snippet, guess, cpu_count())
    end = time.time()
    print(end - start)
    # coreference is almost always true and false pair not same
    # check if wikipedia title helps
    print(data_snippet.head(30))
    # pronouns are either male or female
    print(data_snippet['Pronoun'].value_counts())
    data_snippet = data_snippet[['ID', 'A-guess', 'B-guess']]
    data_snippet.columns = ['ID', 'A-coref', 'B-coref']
    data_snippet.to_csv('CS372_HW5_snippet_output_20150860.tsv', sep='\t', header=False, index=False)
