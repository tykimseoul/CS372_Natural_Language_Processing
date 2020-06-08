import requests
from nltk.stem.snowball import SnowballStemmer
import re
from tabulate import tabulate
import time
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_rows', None)


class TestCase:
    def __init__(self, id, year, organization, sentence, triples):
        self.id = id
        self.year = year
        self.sentence = sentence
        self.organization = organization
        self.triples = triples

    def __iter__(self):
        return iter([self.id, self.year, self.organization])

    def __str__(self):
        return 'TESTCASE\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(self.id, self.year, self.sentence, self.organization, list(map(lambda t: t.to_string(), self.triples)))

    def to_dict(self):
        return {
            'id': self.id,
            'year': self.year,
            'sentence': self.sentence,
            'organization': self.organization,
            'triples': self.triples
        }


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
        dfs.append(df)
        print(df.head())
    full_df = pd.concat(dfs, axis=0)
    full_df.sort_values('sentence', inplace=True)
    print(full_df.describe())
    print(full_df['type'].value_counts())
    return full_df


def collapse_testcases(df):
    testcases = [TestCase(row[0], row[1], row[2], clean_sentence(row[3]), {Triple(row[4], remove_parentheses(row[5]), row[6], remove_parentheses(row[7]))})
                 if row[4] != 0 else TestCase(row[0], row[1], row[2], clean_sentence(row[3]), set())
                 for row in df[['id', 'year', 'org', 'sentence', 'type', 'X', 'action', 'Y']].values]
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
    collapsed_testcases = pd.DataFrame.from_records([t.to_dict() for t in collapsed_testcases])
    return collapsed_testcases


def clip_triple_types(df):
    types = []
    for i in range(6):
        if i == 0:
            type_df = df[df['triples'].apply(lambda t: len(t) == 0)][:20]
            types.append(type_df)
            continue
        type_df = df[df['triples'].apply(lambda t: len(t) > 0 and list(t)[0].type == i)][:20]
        types.append(type_df)
    clipped = pd.concat(types)
    assert len(clipped) == 120
    return clipped


def clean_sentence(sent):
    exclusions = (
        'Aim', 'Background', 'Introduction', 'Objective', 'Aims and objectives', 'Clinical relevance', 'Clinical significance', 'Conclusion', 'Conclusions', 'Impact', 'Main outcome measures',
        'Materials and methods', 'Methods', 'PRACTICAL APPLICATIONS', 'Purpose', 'Results')
    exclusions = tuple(map(lambda s: s + ':', exclusions))
    sent = remove_parentheses(sent)
    sent = re.sub('\\s\\s+', " ", sent.strip())
    if sent.startswith(exclusions):
        idx = sent.index(':')
        sent = sent[idx + 1:]
    sent = sent.strip()
    print(sent)
    return sent


def remove_parentheses(sent):
    return re.sub('\\([^)]*\\)', '', sent)


def test_with(extractor):
    df = read_test_cases(2020, 2014)
    testcases = collapse_testcases(df)
    testcases = clip_triple_types(testcases)
    print(testcases.head(len(testcases)))
    training_testcases = testcases.sample(frac=0.8)
    testing_testcases = testcases.drop(training_testcases.index)
    calculate_performance(extractor, training_testcases)
    calculate_performance(extractor, testing_testcases)


def calculate_performance(extractor, testcases):
    testcases['extractions'] = testcases.apply(lambda t: extractor.extract(t['sentence']), axis=1)
    incorrect = testcases[testcases['triples'] != testcases['extractions']]
    print(incorrect.head(100))
    print('===== RESULT =====\n', len(incorrect), len(testcases))
    true_positive = testcases[(len(testcases['triples']) > 0) & (testcases['triples'] == testcases['extractions'])]
    false_negative = testcases[(len(testcases['triples']) > 0) & (testcases['triples'] != testcases['extractions'])]
    false_positive = testcases[(len(testcases['triples']) == 0) & (testcases['triples'] != testcases['extractions'])]
    # true_negative = testcases[(len(testcases['triples']) == 0) & (testcases['triples'] == testcases['extractions'])]
    precision = len(true_positive) / (len(true_positive) + len(false_positive))
    recall = len(true_positive) / (len(true_positive) + len(false_negative))
    f_score = (2 * precision * recall) / (precision + recall)
    print('Precision: {}\nRecall: {}\nF-score: {}'.format(precision, recall, f_score))
