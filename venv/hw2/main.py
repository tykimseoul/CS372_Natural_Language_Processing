import nltk
from nltk.corpus import brown, wordnet
from collections import defaultdict
from operator import itemgetter
from functools import reduce
from statistics import stdev, StatisticsError
import re
from tabulate import tabulate

# tagged corpus
tagged_corpus = brown.tagged_words()
print("corpus length", len(tagged_corpus))
# create bigrams of the corpus
bigram = list(nltk.bigrams(tagged_corpus))
# dictionary for frequency of each bigram
pairs = defaultdict(lambda: defaultdict(int))
# relevant pos tags
adjectives = ['JJ', 'JJR', 'JJS']
adverbs = ['RB', "RBR", "RBS"]
# irrelevant pos tags
exclusions = ['\.', '\,', "''", ':', '--', 'TO', 'CC', 'IN', 'AT', 'CS', '``', 'MD', 'BE[A-Z]*', 'DT[A-Z]*', 'PPS[A-Z]*']
# join into a single regex
exclusions = "(" + ")|(".join(exclusions) + ")"

# iterate through all bigrams and count frequency of each
for first, second in bigram:
    first = (first[0].lower(), first[1])
    second = (second[0].lower(), second[1])
    if (first[1] in adjectives or first[1] in adverbs) and not re.match(exclusions, second[1]):
        pairs[first][second] += 1

# data structure to convert dictionary into a list of tuples
fixed_data = []
# iterate through frequency of bigrams
for pair in pairs.items():
    # sort in terms of frequency
    words = sorted([(word, count) for word, count in pair[1].items()], key=itemgetter(1), reverse=True)
    # extract frequencies
    counts = list(map(lambda p: p[1], words))
    # if counts.count(counts[0]) == len(counts):
    #     continue
    for w in words:
        # save scores of each bigram
        try:
            fixed_data.append((pair[0], stdev(counts) + w[1], w[0], w[1]))
        except StatisticsError:
            # only a single element in counts
            fixed_data.append((pair[0], w[1], w[0], w[1]))

# sort in terms of score
fixed_data = sorted(fixed_data, key=itemgetter(1), reverse=True)

print(tabulate(fixed_data[:100]))
# for k, s, v in fixed_data[:100]:
#     if s > 0:
#         print("{}\t -> {},\t {}".format(k, s, v))

print("result count: {}".format(len(fixed_data)))

# save to file
f = open('CS372_HW2_output_20150860.csv', 'w')
for t in fixed_data[:100]:
    f.write("{} {}\n".format(t[0][0], t[2][0]))
