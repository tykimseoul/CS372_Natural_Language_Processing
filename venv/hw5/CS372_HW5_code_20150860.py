import pandas as pd
import nltk
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 3000)
pd.set_option('display.max_rows', None)


def word_tokenize_with_dash(sent):
    if '-' in sent:
        sent = re.sub(r"\-", " - ", sent)
        sent = re.sub(r"\s+", " ", sent)
        sent = re.sub(r"\- \-", "--", sent)
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


def read_tsv(filename):
    df = pd.read_csv(filename, sep='\t')
    df['Pronoun-index'] = df.apply(lambda r: get_word_index(r['Text'], r['Pronoun'], r['Pronoun-offset']), axis=1)
    df['A-index'] = df.apply(lambda r: get_word_index(r['Text'], r['A'], r['A-offset']), axis=1)
    df['B-index'] = df.apply(lambda r: get_word_index(r['Text'], r['B'], r['B-offset']), axis=1)
    print(df.head(30))
    invalid = df[df.apply(lambda r: -1 in r['A-index'] or -1 in r['B-index'], axis=1)]
    print(invalid.head(len(invalid)))
    print(len(invalid))
    return df


read_tsv('./GAP/gap-development.tsv')
