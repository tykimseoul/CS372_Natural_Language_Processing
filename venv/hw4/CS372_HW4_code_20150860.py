from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from tabulate import tabulate
import csv
import time


def contains_relevant_verbs(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    sent = list(filter(lambda w: re.match('VB[A-Z]*', w[1]), sent))
    # print(sent)
    stems = list(map(lambda w: stemmer.stem(w[0]), sent))
    relevant = list(filter(lambda s: is_inflection_of(relevant_verbs, s), stems))
    positive = list(filter(lambda s: is_inflection_of(positive_verbs, s), stems))
    negative = list(filter(lambda s: is_inflection_of(negative_verbs, s), stems))
    return len(relevant) + len(positive) + len(negative) > 0


class Article:
    def __init__(self, id, year, abstract, organization):
        self.id = id
        self.year = year
        self.abstract = abstract
        self.relevant_sents = None
        self.organization = organization

    def __iter__(self):
        return iter([self.id, self.year, self.organization])

    def extract_relevant_sentences(self):
        abstract_sents = nltk.sent_tokenize(self.abstract)
        sents = list(filter(lambda s: contains_relevant_verbs(s), abstract_sents))
        self.relevant_sents = sents

    def to_string(self):
        return 'ARTICLE\n\t{}\n\t{}\n\t{}\n\t{}'.format(self.id, self.year, self.relevant_sents, self.organization)


def is_inflection_of(verbs, word):
    verbs = list(map(lambda w: stemmer.stem(w), verbs))
    verbs_regex = "^(" + ")|^(".join(verbs) + ")"
    return re.match(verbs_regex, word)


relevant_verbs = ['activate', 'inhibit', 'bind']
positive_verbs = ['accelerate', 'augment', 'induce', 'stimulate', 'require', 'up-regulate']
negative_verbs = ['abolish', 'block', 'down-regulate', 'prevent']

stemmer = SnowballStemmer("english")

medline_url = 'https://pubmed.ncbi.nlm.nih.gov/'
crawl_per_year = 500
params = {
    'size': 50,
    'page': 1
}
crawling_year = 2020
articles = []


def crawl_abstract(id, year):
    try:
        html = requests.get(medline_url + id)
    except requests.exceptions.ConnectionError as e:
        print('pausing..', e)
        time.sleep(10)
        html = requests.get(medline_url + id)
    soup = BeautifulSoup(html.text, "html.parser")
    abstract_content = soup.find(class_='abstract-content')
    if abstract_content is None:
        return None
    organizations = soup.find(class_='item-list')
    if organizations is None:
        return None
    organizations = organizations.findAll('li')
    try:
        organization = organizations[-1].contents[1].strip()
    except TypeError:
        return None
    atc = Article(id, year, abstract_content.text, organization)
    atc.extract_relevant_sentences()
    print(atc.to_string())
    return atc


def search_year(year):
    term = '(English[Language]) AND (("{}"[Date - Publication] : "{}"[Date - Publication]))'.format(year, year)
    params['term'] = term
    relevant_articles = []
    page = 0
    while len(relevant_articles) < 60:
        page += 1
        params['page'] = page
        html = requests.get(medline_url, params=params)
        soup = BeautifulSoup(html.text, "html.parser")
        results = soup.find_all('a', attrs={'class': 'labs-docsum-title'})
        links = list(map(lambda r: r['href'].split('/')[1], results))
        print(links)
        year_articles = list(map(lambda l: crawl_abstract(l, year), links))
        year_articles = list(filter(lambda a: a is not None and len(a.relevant_sents) > 0, year_articles))
        print('year {} page {}: {}'.format(year, page, len(year_articles)))
        relevant_articles.extend(year_articles)
    return relevant_articles


while len(articles) < 300:
    year_articles = search_year(crawling_year)
    articles.extend(year_articles)
    with open('corpus_untagged_{}.csv'.format(crawling_year), 'w') as csv_file:
        wr = csv.writer(csv_file)
        for article in year_articles:
            for i in article.relevant_sents:
                wr.writerow(list(article) + [i])
    crawling_year -= 1
