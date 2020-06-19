import pandas as pd
import nltk
import re
from itertools import accumulate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 3000)
pd.set_option('display.max_rows', None)

exclusions = ['JJ.?', 'RB.?']
# join into a single regex
exclusions = "(" + ")|(".join(exclusions) + ")"


def word_tokenize_with_dash(sent):
    # if '-' in sent:
    #     sent = re.sub(r"-", " - ", sent)
    #     sent = re.sub(r"\s+", " ", sent)
    #     sent = re.sub(r"- -", "--", sent)
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
    sent_index = lengths_acc.index(next(filter(lambda l: l > offset, lengths_acc))) - 1
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
    df['Text-labeled'] = df.apply(lambda r: label_entities(r['Text-simplified'], r['Pronoun-sent-index-simplified'], r['A-sent-index-simplified'], r['B-sent-index-simplified']), axis=1)
    df['reduction'] = df.apply(lambda r: '{} -> {}'.format(len(nltk.sent_tokenize(r['Text'])), len(nltk.sent_tokenize(r['Text-simplified']))), axis=1)
    return df


data = read_tsv('./GAP/gap-development.tsv')[:30]
data = find_indices(data)
data = simplify(data)
print(data.head(30))
