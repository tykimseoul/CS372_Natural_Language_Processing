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

# cache for male and female names
all_male_names = defaultdict(set)
all_female_names = defaultdict(set)


# custom word tokenizing function
def word_tokenize_with_dash(sent):
    sent = re.sub(r"\.,", ",", sent)
    return nltk.word_tokenize(sent)


# find the word index of an entity within the sentence given the offset
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


# find the index of the sentence the entity belongs to
def get_sent_index(text, entity, offset=0):
    sents = nltk.sent_tokenize(text)
    lengths = list(map(len, sents))
    lengths_acc = [0] + list(accumulate(lengths))
    sent_index = lengths_acc.index(next(filter(lambda l: l >= offset, lengths_acc))) - 1
    offset -= (lengths_acc[sent_index] + sent_index)
    word_index = get_word_index(sents[sent_index], entity, offset)
    return sent_index, word_index


# remove sentences that does not contain the pronoun, A or B
def reduce_text(text, relevant_indices):
    sents = nltk.sent_tokenize(text)
    relevant_sents = list(filter(lambda s: s[0] in relevant_indices, enumerate(sents)))
    relevant_sents = list(map(lambda s: s[1], relevant_sents))
    return ' '.join(relevant_sents)


# find indices of sentences that contain the pronoun, A or B
def relevant_sentences(text, pronoun, a, b):
    indices = [pronoun, a, b]
    sents = nltk.sent_tokenize(text)
    lengths = list(map(len, sents))
    lengths_acc = [0] + list(accumulate(lengths))
    relevant_indices = set(map(lambda i: lengths_acc.index(next(filter(lambda l: l > i, lengths_acc))) - 1, indices))
    return relevant_indices


# find the index of the simplified sentence the entity belongs to
def get_simplified_sent_index(text, sent_index, relevant_indices):
    irrelevant_indices = set(range(len(nltk.sent_tokenize(text)))) - relevant_indices
    irrelevant_count_before_current = len(set(filter(lambda i: i < sent_index[0], irrelevant_indices)))
    return sent_index[0] - irrelevant_count_before_current, sent_index[1]


# read tsv file into a dataframe
def read_tsv(filename):
    df = pd.read_csv(filename, sep='\t')
    return df


# calculate word and sentence indices of the pronoun, A and B
def find_indices(df):
    df['Pronoun-index'] = df.apply(lambda r: get_word_index(r['Text'], r['Pronoun'], r['Pronoun-offset']), axis=1)
    df['A-index'] = df.apply(lambda r: get_word_index(r['Text'], r['A'], r['A-offset']), axis=1)
    df['B-index'] = df.apply(lambda r: get_word_index(r['Text'], r['B'], r['B-offset']), axis=1)
    df['Pronoun-sent-index'] = df.apply(lambda r: get_sent_index(r['Text'], r['Pronoun'], r['Pronoun-offset']), axis=1)
    df['A-sent-index'] = df.apply(lambda r: get_sent_index(r['Text'], r['A'], r['A-offset']), axis=1)
    df['B-sent-index'] = df.apply(lambda r: get_sent_index(r['Text'], r['B'], r['B-offset']), axis=1)
    return df

# simplify text to contain only relevant sentences and recalculate sentence indices
def simplify(df):
    df['Relevant-sentences'] = df.apply(lambda r: relevant_sentences(r['Text'], r['Pronoun-offset'], r['A-offset'], r['B-offset']), axis=1)
    df['Text-simplified'] = df.apply(lambda r: reduce_text(r['Text'], r['Relevant-sentences']), axis=1)
    df['Pronoun-sent-index-simplified'] = df.apply(lambda r: get_simplified_sent_index(r['Text'], r['Pronoun-sent-index'], r['Relevant-sentences']), axis=1)
    df['A-sent-index-simplified'] = df.apply(lambda r: get_simplified_sent_index(r['Text'], r['A-sent-index'], r['Relevant-sentences']), axis=1)
    df['B-sent-index-simplified'] = df.apply(lambda r: get_simplified_sent_index(r['Text'], r['B-sent-index'], r['Relevant-sentences']), axis=1)
    df['Sent-simplified-lengths'] = df.apply(lambda r: [len(word_tokenize_with_dash(s)) for s in nltk.sent_tokenize(r['Text-simplified'])], axis=1)
    return df


# extract candidates for the entity in the snippet context
def extract_candidates_snippet_context(sent, pronoun):
    sents = nltk.sent_tokenize(sent)
    # ignore those that appear in a different sentence after one that contains the pronoun
    sents = sents[:pronoun[0] + 1]
    print(sents)
    if len(sents) == 0:
        print("WHY!!!!!!!!", sent)
        return []

    def single_sent_candidates(sent):
        raw_sent = sent
        untagged_sent = word_tokenize_with_dash(sent)
        sent = nltk.pos_tag(untagged_sent)
        cp = nltk.RegexpParser('NNP: {<NNP>+}')
        sent = cp.parse(sent)
        # collect only NNP phrases
        candidate = list(filter(lambda s: type(s) == nltk.tree.Tree and s.label() == 'NNP', sent))
        candidate = list(map(lambda t: t.flatten(), candidate))
        candidate = list(map(lambda t: t.leaves(), candidate))
        candidate = list(map(lambda l: ' '.join([p[0] for p in l]), candidate))
        # check Wikipedia for type and gender of the entity
        genders = list(map(lambda e: check_wikipedia(e), candidate))
        positions = []
        for c in candidate:
            try:
                p = list(nltk.ngrams(untagged_sent, len(word_tokenize_with_dash(c)))).index(tuple(word_tokenize_with_dash(c)))
            except ValueError:
                p = raw_sent.find(c)
            positions.append(p)
        return list(zip(candidate, genders, positions))

    candidates = list(map(lambda s: single_sent_candidates(s), sents))
    result = []
    for s in enumerate(candidates):
        indexed = [(c[0], c[1], s[0], c[2]) for c in s[1]]
        result.extend(indexed)
    # remove entities that are not people
    result = list(filter(lambda r: r[1][0] != 'NOT', result))
    return result


# keep only alphanumeric characters from a string
def alphanumeric(s):
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\[[^\]]*\]', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return re.sub(r'[^A-Za-z0-9 ]+', '', s)


# extract candidates for the entity in the page context
def extract_candidates_page_context(sent, pronoun, url):
    sents = nltk.sent_tokenize(sent)
    sents = sents[:pronoun[0] + 1]
    print(sents)
    if len(sents) == 0:
        print("WHY!!!!!!!!", sent)
        return []
    html = safe_request(url)
    soup = BeautifulSoup(html.text, "html.parser")
    try:
        content = soup.find(class_='mw-parser-output').findAll('p')
    except AttributeError:
        content = soup.find(id='mw-content-text').findAll('p')
    content_text = list(map(lambda c: c.text, content))
    try:
        star_idx = sents[0].index('*')
    except ValueError:
        star_idx = 1000000
    content_sents = [[alphanumeric(c) for c in nltk.sent_tokenize(t)] for t in content_text]
    content_idx = [any([alphanumeric(sents[0][:min(star_idx, 20)]) in t for t in c]) for c in content_sents]
    if not any(content_idx):
        return []
    # find where the given sentence is located within the page
    idx = content_idx.index(True)
    # ignore those that appear in a different sentence after one that contains the pronoun
    content = content[:idx + 1]
    # get items that are Wikipedia entries
    links = list(map(lambda c: c.findAll('a'), content))
    links = list(map(lambda l: [k.text for k in l], links))
    links = [item for sublist in links for item in sublist]
    links = list(filter(lambda l: not re.match(r'\[.+\]', l), links))
    links.append(soup.find(class_='firstHeading').text)
    # take only closest 10 terms
    links = links[-10:]
    # check Wikipedia for type and gender of the entity
    links = list(map(lambda l: (l, check_wikipedia(l)), links))
    # remove entities that are not people
    links = list(filter(lambda c: c[1][0] != 'NOT', links))
    print(len(links), links)
    return links


# a safe function against connection error
def safe_request(link):
    try:
        html = requests.get(link)
    except requests.exceptions.ConnectionError as e:
        print('pausing..', e)
        time.sleep(5)
        html = requests.get(link)
    return html


# check Wikipedia for the type and gender of the provided entity
def check_wikipedia(entity):
    html = safe_request('https://en.wikipedia.org/wiki/{}'.format(entity))
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
            # fallback function
            if gender == 'UNK':
                content = soup.find(class_='mw-parser-output').find('p').text
                content = list(map(lambda w: w.lower(), word_tokenize_with_dash(content)))
                if 'he' in content or 'him' in content or 'his' in content:
                    gender = 'M'
                elif 'she' in content or 'her' in content:
                    gender = 'F'
    result = (person, gender)
    return result


# determine gender of a name from a list of male and female names
def determine_gender(name):
    name = word_tokenize_with_dash(name)
    # can determine the first name only if more than one term is provided
    if len(name) > 1:
        is_male = gender_helper('https://en.wikipedia.org/w/index.php?title=Category:English_masculine_given_names&from={}'.format(name[0][0]), name[0], 0)
        is_female = gender_helper('https://en.wikipedia.org/w/index.php?title=Category:English_feminine_given_names&from={}'.format(name[0][0]), name[0], 1)
        if is_male != is_female:
            if is_male:
                return 'M'
            else:
                return 'F'
    return 'UNK'


# helper function for determining gender
def gender_helper(link, name, gender):
    all_names = {0: all_male_names, 1: all_female_names}
    first_letter = name[0].lower()
    if len(all_names[gender][first_letter]) == 0:
        html = safe_request(link)
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


# calculate the distance between candidates and the pronoun
def calculate_distance(lengths, pronoun, pronoun_index, candidates):
    def dist(pronoun_sent_idx, pronoun_word_idx, word_sent_idx, word_idx):
        if pronoun_sent_idx < word_sent_idx:
            return lengths[pronoun_sent_idx] - pronoun_word_idx + word_idx
        elif pronoun_sent_idx > word_sent_idx:
            return lengths[word_sent_idx] - word_idx + pronoun_word_idx
        else:
            return abs(pronoun_word_idx - word_idx)

    distances = list(map(lambda c: dist(pronoun_index[0], pronoun_index[1][0], c[2], c[3]), candidates))
    if pronoun.lower() == 'he' or pronoun.lower() == 'his' or pronoun.lower() == 'him':
        wrong_gender = list(map(lambda c: c[0] if c[1][1][1] == 'F' else -1, enumerate(candidates)))
    elif pronoun.lower() == 'she' or pronoun.lower() == 'her':
        wrong_gender = list(map(lambda c: c[0] if c[1][1][1] == 'M' else -1, enumerate(candidates)))
    else:
        wrong_gender = []
    # assign high distance to the wrong gender
    distances = list(map(lambda d: 100000 if d[0] in wrong_gender else d[1], enumerate(distances)))
    return distances


# extract candidates in snippet context
def extract_snippet_context(df):
    df['candidates'] = df.apply(lambda r: extract_candidates_snippet_context(r['Text-simplified'], r['Pronoun-sent-index-simplified']), axis=1)
    return df


# extract candidates in page context
def extract_page_context(df):
    df['candidates'] = df.apply(lambda r: extract_candidates_page_context(r['Text-simplified'], r['Pronoun-sent-index-simplified'], r['URL']), axis=1)
    return df


# choose the best candidate in the snippet context
def choose_candidate_snippet_context(a, b, candidates, distances):
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

    # if both A and B are not present in candidates
    if a_idx == -1 and b_idx == -1:
        return a_guess, b_guess
    # if both A and B are not the closest candidate
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


# choose the best candidate in the page context
def choose_candidate_page_context(pronoun, a, b, candidates):
    a_guess = False
    b_guess = False
    a = a.lower()
    b = b.lower()
    # filter only candidates with correct gender
    if pronoun.lower() == 'he' or pronoun.lower() == 'his' or pronoun.lower() == 'him':
        candidates = list(filter(lambda c: c[1][1] == 'M' or c[1][1] == 'UNK', candidates))
    elif pronoun.lower() == 'she' or pronoun.lower() == 'her':
        candidates = list(filter(lambda c: c[1][1] == 'F' or c[1][1] == 'UNK', candidates))
    candidates = list(map(lambda c: c[0].lower(), candidates))
    a_idx = list(map(lambda c: tuple(word_tokenize_with_dash(a)) in list(nltk.ngrams(word_tokenize_with_dash(c), len(word_tokenize_with_dash(a)))), candidates))
    b_idx = list(map(lambda c: tuple(word_tokenize_with_dash(b)) in list(nltk.ngrams(word_tokenize_with_dash(c), len(word_tokenize_with_dash(b)))), candidates))
    if True in a_idx:
        a_idx = len(a_idx) - a_idx[::-1].index(True) - 1
    else:
        a_idx = -1
    if True in b_idx:
        b_idx = len(b_idx) - b_idx[::-1].index(True) - 1
    else:
        b_idx = -1
    # if both A and B are not candidates
    if a_idx == -1 and b_idx == -1:
        return a_guess, b_guess
    else:
        a_guess = a_idx > b_idx
        b_guess = a_idx < b_idx
        return a_guess, b_guess


# determine the coreference in snippet context case
def guess_snippet_context(df):
    df['distance'] = df.apply(lambda r: calculate_distance(r['Sent-simplified-lengths'], r['Pronoun'], r['Pronoun-sent-index-simplified'], r['candidates']), axis=1)
    df['guess'] = df.apply(lambda r: choose_candidate_snippet_context(r['A'], r['B'], r['candidates'], r['distance']), axis=1)
    df['A-guess'], df['B-guess'] = zip(*df.guess)
    df.drop('guess', axis=1, inplace=True)
    return df


# determine the coreference in page context case
def guess_page_context(df):
    df['guess'] = df.apply(lambda r: choose_candidate_page_context(r['Pronoun'], r['A'], r['B'], r['candidates']), axis=1)
    df['A-guess'], df['B-guess'] = zip(*df.guess)
    df.drop('guess', axis=1, inplace=True)
    return df


# use multiprocessing, otherwise it takes too long
def parallelize(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    df_split = list(filter(lambda d: len(d) > 0, df_split))
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# run snippet context case
def do_snippet_context(data):
    data_snippet = parallelize(data, extract_snippet_context, min(10, cpu_count()))
    data_snippet = parallelize(data_snippet, guess_snippet_context, cpu_count())
    return data_snippet


# run page context case
def do_page_context(data):
    data_page = parallelize(data, extract_page_context, min(20, cpu_count()))
    data_page = parallelize(data_page, guess_page_context, cpu_count())
    return data_page


if __name__ == '__main__':
    set_start_method("spawn")
    data = read_tsv('./GAP/gap-test.tsv')
    start = time.time()
    data = parallelize(data, find_indices, cpu_count())
    data = parallelize(data, simplify, cpu_count())
    data = do_page_context(data)
    end = time.time()
    print(end - start)
    # coreference is almost always a true and false pair not same
    print(data.head(30))
    data = data[['ID', 'A-guess', 'B-guess']]
    data.columns = ['ID', 'A-coref', 'B-coref']
    data.to_csv('CS372_HW5_page_output_20150860.tsv', sep='\t', header=False, index=False)
