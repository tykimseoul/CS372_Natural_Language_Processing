import nltk
from nltk.corpus import brown, wordnet
from collections import defaultdict
from operator import itemgetter
from functools import reduce
from statistics import stdev

tagged_corpus = brown.tagged_words(categories=["news", "editorial"])
print("corpus length", len(tagged_corpus))
bigram = list(nltk.bigrams(tagged_corpus))
pairs = defaultdict(lambda: defaultdict(int))
adjectives = ['JJ', 'JJR', 'JJS']

for first, second in bigram:
    if first[1] in adjectives:
        pairs[first][second] += 1

fixed_data = []
for pair in pairs.items():
    words = [(word, count) for word, count in pair[1].items()]
    counts = list(map(lambda p: p[1], words))
    if counts.count(counts[0]) == len(counts):
        continue
    fixed_data.append((pair[0], stdev(counts), sorted(words, key=itemgetter(1), reverse=True)))

fixed_data = sorted(fixed_data, key=itemgetter(1), reverse=True)

for k, s, v in fixed_data:
    print("{} -> {}, {}".format(k, s, v))
