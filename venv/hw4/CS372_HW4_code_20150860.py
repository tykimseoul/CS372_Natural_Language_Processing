from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from tabulate import tabulate


def contains_relevant_verbs(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    sent = list(filter(lambda w: re.match('VB[A-Z]*', w[1]), sent))
    print(sent)
    stems = list(map(lambda w: stemmer.stem(w[0]), sent))
    relevant = list(filter(lambda s: is_inflection_of(relevant_verbs, s), stems))
    positive = list(filter(lambda s: is_inflection_of(positive_verbs, s), stems))
    negative = list(filter(lambda s: is_inflection_of(negative_verbs, s), stems))
    print(relevant, positive, negative)
    return len(relevant) * len(positive) * len(negative) > 0


class Article:
    def __init__(self, date, abstract, organizations):
        self.date = date
        self.abstract = abstract
        self.relevant_sents = None
        self.organizations = organizations

    def extract_relevant_sentences(self):
        abstract_sents = nltk.sent_tokenize(self.abstract)
        sents = list(filter(lambda s: contains_relevant_verbs(s), abstract_sents))
        self.relevant_sents = sents

    def to_string(self):
        return 'ARTICLE\n\t{}\n\t{}\n\t{}'.format(self.date, self.relevant_sents, self.organizations)


def is_inflection_of(verbs, word):
    verbs = list(map(lambda w: stemmer.stem(w), verbs))
    verbs_regex = "^(" + ")|^(".join(verbs) + ")"
    return re.match(verbs_regex, word)


relevant_verbs = ['activate', 'inhibit', 'bind']
positive_verbs = ['accelerate', 'augment', 'induce', 'stimulate', 'require', 'up-regulate']
negative_verbs = ['abolish', 'block', 'down-regulate', 'prevent']

stemmer = SnowballStemmer("english")

medline_url = 'https://pubmed.ncbi.nlm.nih.gov/'

html = requests.get(medline_url + '29545796')
soup = BeautifulSoup(html.text, "html.parser")
abstract_content = soup.find(class_='abstract-content')
date = soup.find(class_='cit').text.split(';')[0]
organizations = soup.find(class_='item-list').findAll('li')
organizations = dict(map(lambda o: (int(o.contents[0].text), o.contents[1].strip()), organizations))
abstract_sents = nltk.sent_tokenize(abstract_content.text)
atc = Article(date, abstract_content.text, organizations)
atc.extract_relevant_sentences()
print(atc.to_string())
