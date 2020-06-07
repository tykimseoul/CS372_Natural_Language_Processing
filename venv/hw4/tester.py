import requests
from nltk.stem.snowball import SnowballStemmer
import re
from tabulate import tabulate
import time
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class TestCase:
    def __init__(self, id, year, organization, sentence, triples):
        self.id = id
        self.year = year
        self.sentence = sentence
        self.organization = organization
        self.triples = triples

    def __iter__(self):
        return iter([self.id, self.year, self.organization])

    def to_string(self):
        return 'TESTCASE\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(self.id, self.year, self.sentence, self.organization, list(map(lambda t: t.to_string(), self.triples)))


class Triple:
    def __init__(self, type=None, x=None, action=None, y=None):
        self.type = type
        self.x = x
        self.action = action
        self.y = y

    def __str__(self):
        return 'Triple({}, {}, {}, {})'.format(self.type, self.x, self.action, self.y)

    def __eq__(self, other):
        if isinstance(other, Triple):
            return self.x == other.x and self.action == other.action and self.y == other.y
        return False

    def __hash__(self):
        return hash(tuple((self.x, self.action, self.y)))


def read_test_cases(start, end):
    dfs = []
    for year in range(start, end - 1, -1):
        df = pd.read_csv('corpus_tagged_{}.csv'.format(year), encoding='unicode_escape')
        df.drop(df.columns[[8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
        df.drop(df[df.type == -1].index, inplace=True)
        df.drop(df[df.type == 0].index, inplace=True)
        dfs.append(df)
        print(df.head())
    full_df = pd.concat(dfs, axis=0)
    full_df.sort_values('sentence', inplace=True)
    print(full_df.describe())
    print(full_df['type'].value_counts())
    return full_df


def collapse_testcases(df):
    testcases = [TestCase(row[0], row[1], row[2], clean_sentence(row[3]), {Triple(row[4], row[5], row[6], row[7])}) for row in
                 df[['id', 'year', 'org', 'sentence', 'type', 'X', 'action', 'Y']].values]
    collapse_index = 0
    collapsed_testcases = []
    for idx, t in enumerate(testcases):
        if idx == collapse_index:
            collapsed_testcases.append(t)
        elif collapsed_testcases[collapse_index].sentence == t.sentence:
            collapsed_testcases[collapse_index].triples.update(t.triples)
        elif collapsed_testcases[collapse_index].sentence != t.sentence:
            collapse_index += 1
            collapsed_testcases.append(t)
    print("testcase size:", len(collapsed_testcases))
    assert df['sentence'].nunique() == len(collapsed_testcases)
    return collapsed_testcases


def clean_sentence(sent):
    exclusions = (
        'Aim', 'Background', 'Introduction', 'Objective', 'Aims and objectives', 'Clinical relevance', 'Clinical significance', 'Conclusion', 'Conclusions', 'Impact', 'Main outcome measures',
        'Materials and methods', 'Methods', 'PRACTICAL APPLICATIONS', 'Purpose', 'Results')
    exclusions = tuple(map(lambda s: s + ':', exclusions))
    sent = re.sub('\\([^)]*\\)', '', sent)
    sent = re.sub('\\s\\s+', " ", sent.strip())
    if sent.startswith(exclusions):
        idx = sent.index(':')
        sent = sent[idx + 1:]
    sent = sent.strip()
    print(sent)
    return sent


def test_with(extractor):
    df = read_test_cases(2020, 2014)
    testcases = collapse_testcases(df)
    sentences = list(map(lambda t: t.sentence, testcases))
    relations = list(map(lambda t: t.triples, testcases))
    extractions = list(map(lambda t: extractor.extract(t.sentence), testcases))
    test_size = len(testcases)
    assert [len(sentences), len(relations), len(extractions)] == [test_size, test_size, test_size]
    df = pd.DataFrame({'sentence': sentences, 'relations': relations, 'extractions': extractions})
    incorrect = df[df['relations'] != df['extractions']]
    print(incorrect.head(100))
    print('===== RESULT =====\n', len(incorrect), len(df))
