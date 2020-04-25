import nltk
from nltk.corpus import brown, wordnet
from collections import defaultdict
from operator import itemgetter
from functools import reduce
from statistics import stdev, StatisticsError
import re
from tabulate import tabulate

tagged_corpus = brown.tagged_words()
print("corpus length", len(tagged_corpus))
bigram = list(nltk.bigrams(tagged_corpus))
pairs = defaultdict(lambda: defaultdict(int))
adjectives = ['JJ', 'JJR', 'JJS']
adverbs = ['RB', "RBR", "RBS"]
exclusions = ['\.', '\,', "''", ':', '--', 'TO', 'CC', 'IN', 'AT', 'CS', '``', 'MD', 'BE[A-Z]*', 'DT[A-Z]*', 'PPS[A-Z]*']
exclusions = "(" + ")|(".join(exclusions) + ")"

for first, second in bigram:
    first = (first[0].lower(), first[1])
    second = (second[0].lower(), second[1])
    if (first[1] in adjectives or first[1] in adverbs) and not re.match(exclusions, second[1]):
        pairs[first][second] += 1

fixed_data = []
for pair in pairs.items():
    words = sorted([(word, count) for word, count in pair[1].items()], key=itemgetter(1), reverse=True)
    counts = list(map(lambda p: p[1], words))
    # if counts.count(counts[0]) == len(counts):
    #     continue
    for w in words:
        try:
            fixed_data.append((pair[0], stdev(counts) + w[1], w[0], w[1]))
        except StatisticsError:
            fixed_data.append((pair[0], w[1], w[0], w[1]))

fixed_data = sorted(fixed_data, key=itemgetter(1), reverse=True)

print(tabulate(fixed_data[:100]))
# for k, s, v in fixed_data[:100]:
#     if s > 0:
#         print("{}\t -> {},\t {}".format(k, s, v))

print("result count: {}".format(len(fixed_data)))

f = open('CS372_HW2_output_20150860.csv', 'w')
for t in fixed_data[:100]:
    f.write("{} {}\n".format(t[0][0], t[2][0]))
