from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import csv
import time

'''determine if the provided sentence contains relevant verbs (activate, inhibit etc)'''
def contains_relevant_verbs(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    sent = list(filter(lambda w: re.match('VB[A-Z]*', w[1]), sent))
    stems = list(map(lambda w: stemmer.stem(w[0]), sent))
    relevant = list(filter(lambda s: is_inflection_of(relevant_verbs, s), stems))
    positive = list(filter(lambda s: is_inflection_of(positive_verbs, s), stems))
    negative = list(filter(lambda s: is_inflection_of(negative_verbs, s), stems))
    return len(relevant) + len(positive) + len(negative) > 0

'''
Class definition for Article class.
Encapsulates the id, year, abstract and organization of an article.
'''
class Article:
    def __init__(self, id, year, abstract, organization):
        self.id = id
        self.year = year
        self.abstract = abstract
        self.relevant_sents = None
        self.organization = organization

    def __iter__(self):
        return iter([self.id, self.year, self.organization])

    # filter relevant sentences from all sentences of the abstract.
    def extract_relevant_sentences(self):
        abstract_sents = nltk.sent_tokenize(self.abstract)
        sents = list(filter(lambda s: contains_relevant_verbs(s), abstract_sents))
        self.relevant_sents = sents

    def to_string(self):
        return 'ARTICLE\n\t{}\n\t{}\n\t{}\n\t{}'.format(self.id, self.year, self.relevant_sents, self.organization)


'''determine of the provided word is an inflection of one of verbs in verbs list.'''
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
'''crawling parameters'''
params = {
    'size': 50,
    'page': 1
}
crawling_year = 2020
articles = []


'''crawl abstract of article with the provided id'''
def crawl_abstract(id, year):
    try:
        html = requests.get(medline_url + id)
    except requests.exceptions.ConnectionError as e:
        # the server hates being crawled too frequently
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
    # create article object
    atc = Article(id, year, abstract_content.text, organization)
    atc.extract_relevant_sentences()
    print(atc.to_string())
    return atc


'''search all articles from the provided year'''
def search_year(year):
    term = '(English[Language]) AND (("{}"[Date - Publication] : "{}"[Date - Publication]))'.format(year, year)
    params['term'] = term
    relevant_articles = []
    page = 0
    # crawl 60 articles from each year (arbitrarily determined)
    while len(relevant_articles) < 60:
        # increment search result pages
        page += 1
        params['page'] = page
        html = requests.get(medline_url, params=params)
        soup = BeautifulSoup(html.text, "html.parser")
        results = soup.find_all('a', attrs={'class': 'labs-docsum-title'})
        # all links in this page
        links = list(map(lambda r: r['href'].split('/')[1], results))
        print(links)
        # all articles in this page
        year_articles = list(map(lambda l: crawl_abstract(l, year), links))
        year_articles = list(filter(lambda a: a is not None and len(a.relevant_sents) > 0, year_articles))
        print('year {} page {}: {}'.format(year, page, len(year_articles)))
        relevant_articles.extend(year_articles)
    return relevant_articles


'''crawl title of the article with the provided id'''
def crawl_title(id):
    id = str(id)
    try:
        html = requests.get(medline_url + id)
    except requests.exceptions.ConnectionError as e:
        print('pausing..', e)
        time.sleep(10)
        html = requests.get(medline_url + id)
    soup = BeautifulSoup(html.text, "html.parser")
    title = soup.find(class_='heading-title').text.strip()
    return title


'''main function that crawls articles that contain relevant sentences in their abstracts.'''
if __name__ == "__main__":
    while len(articles) < 300:
        year_articles = search_year(crawling_year)
        articles.extend(year_articles)
        with open('corpus_untagged_{}.csv'.format(crawling_year), 'w') as csv_file:
            # save in a csv file
            wr = csv.writer(csv_file)
            for article in year_articles:
                for i in article.relevant_sents:
                    wr.writerow(list(article) + [i.strip()])
        crawling_year -= 1
