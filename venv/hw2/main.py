import nltk
from nltk.corpus import brown, wordnet
from collections import defaultdict
from operator import itemgetter
from functools import reduce

tagged_corpus = brown.tagged_words(categories=["news", "editorial"])
print("corpus length", len(tagged_corpus))
bigram = list(nltk.bigrams(tagged_corpus))
pairs = defaultdict(lambda: defaultdict(int))
adjectives = ['JJ', 'JJR', 'JJS']

for first, second in bigram:
    if first[1] in adjectives:
        pairs[first][second] += 1

fixed_data = defaultdict(list)
for pair in pairs.items():
    words = [(word, count) for word, count in pair[1].items()]
    counts = list(map(lambda p: p[1], words))
    if counts.count(counts[0]) == len(counts):
        continue
    fixed_data[pair[0]] = sorted(words, key=itemgetter(1), reverse=True)

for k, v in fixed_data.items():
    print("{} -> {}".format(k, v))
