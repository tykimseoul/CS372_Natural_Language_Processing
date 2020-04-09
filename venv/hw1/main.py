import nltk
from nltk.corpus import brown, wordnet
import itertools
from math import floor
from nltk.stem.snowball import SnowballStemmer


# find all unique occurrences of pos in corpus
def all_occurrences(pos, corpus):
    occurrences = list(filter(lambda a: a[1] in pos, corpus))
    occurrences = list(map(lambda a: a[0], occurrences))
    return list(dict.fromkeys(occurrences))


# create a frequency map of word of give pos
def word_counts(pos, corpus):
    result = {}
    occurrences = list(filter(lambda a: a[1] in pos, corpus))
    for i in occurrences:
        result[i[0].lower()] = result.get(i[0].lower(), 0) + 1
    return result


# generate intensity modifying adverbs from seed adverbs by combining adverb synonyms of each
def adverbs_of_degree():
    seeds = ["extremely", "quite", "just", "almost", "very", "too", "enough", "slightly", "completely"]
    # get adverb synonyms of each of seed adverbs
    generated = list(map(lambda s: wordnet.synsets(s, pos=wordnet.ADV), seeds))
    # flatten the list of lists
    generated = [item.lemma_names()[0] for sublist in generated for item in sublist]
    # turn into a set to remove duplicates
    generated = set(generated)
    # include the seed adverbs
    generated = generated | set(seeds)
    # filter only those found in the corpus
    generated = set(filter(lambda g: g in raw_corpus, generated))
    print(generated)
    return generated


stemmer = SnowballStemmer("english")
# corpus of a specific category with pos tags
tagged_corpus = brown.tagged_words(categories=["news", "editorial"])
raw_corpus = brown.words(categories=["news", "editorial"])
print("corpus length", len(tagged_corpus))
# pos tags for each pos
verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjectives = ['JJ', 'JJR', 'JJS']
adverbs = ['RB', "RBR", "RBS"]
# getting intensity modifying adverbs
all_adverbs = adverbs_of_degree()
# finding occurrences of verbs and adjectives
all_verbs = all_occurrences(verbs, tagged_corpus)
all_adjectives = all_occurrences(adjectives, tagged_corpus)

# counts of all verbs and adjectives
verb_counts = word_counts(verbs, tagged_corpus)
adjective_counts = word_counts(adjectives, tagged_corpus)

# dictionary of synsets of each verb and adjective
all_verb_synsets = dict(map(lambda s: (s.lower(), wordnet.synsets(s, pos=wordnet.VERB)), all_verbs))
all_adjective_synsets = dict(map(lambda s: (s.lower(), wordnet.synsets(s, pos=wordnet.ADJ)), all_adjectives))
# empty lists of adverb-verb and adverb-adjective pairs
adverb_verb = []
adverb_adjective = []


# zips a list with the next element into a list of tuples
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# find words with maximum similarity to the word given pos
def maximum_similarity(word, target_synsets, pos):
    # synset of given word
    word_synset = wordnet.synsets(word, pos=pos)
    word = word.lower()
    if pos == wordnet.VERB:
        word_count = verb_counts[word]
    else:
        word_count = adjective_counts[word]
    # stem of given word
    word_stem = stemmer.stem(word)
    similarities = []
    # for each key of given synsets
    for target in target_synsets:
        target = target.lower()
        # stem of target word
        target_stem = stemmer.stem(target)
        # if given word and target word have different stems (i.e. not different tenses of the same verb etc.)
        if word_stem != target_stem:
            if pos == wordnet.VERB:
                target_count = verb_counts[target]
            else:
                target_count = adjective_counts[target]
            # compare frequency of word and target
            if word_count > target_count:
                # calculate similarity between synsets of two words
                sim = synset_similarity(word_synset, target_synsets[target])
                # record the similarity score
                similarities.append((target, sim, stemmer.stem(target)))

    # sort the similarity by the score in descending order and get top 50
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:50]
    # get the set of unique stems of similar words
    stems = set(map(lambda t: t[2], similarities))
    filtered = []
    # loop through similar words to ensure that words of same stem are included only once
    for sim in similarities:
        if len(stems) == 0:
            break
        if sim[2] in stems:
            filtered.append((sim[0], sim[1]))
            # filtered.append(sim)
            stems.remove(sim[2])
    # return top 5 most similar words
    return filtered[:5]


# calculate similarity score between two synsets
def synset_similarity(first_set, second_set):
    # how much of each set to include in the calculation
    limit = 0.3
    # cannot calculate similarity between empty sets
    if len(first_set) == 0 or len(second_set) == 0:
        return 0
    scores = []
    # nested loop across two sets limited by the limit value
    for f in first_set[:max(floor(len(first_set) * limit), 1)]:
        for s in second_set[:max(floor(len(second_set) * limit), 1)]:
            # calculate path_similarity between two words
            score = safe_similarity(f, s)
            # record the score
            scores.append(score)
    # return the average of the scores
    return sum(scores) / len(scores)


# calculate similarity between two words safe against one-way path
def safe_similarity(f, s):
    # calculate path similarity in both ways
    sim1 = f.path_similarity(s)
    sim2 = s.path_similarity(f)
    # compare the two similarities
    if sim1 != sim2:
        # if different, return the non-None value
        if sim1 is None:
            return sim2
        else:
            return sim1
    else:
        # if same
        if sim1 is None:
            # if both None, similarity is 0
            return 0
        else:
            # if both valid, return it
            return sim1


# for each pairs of words
for first, second in pairwise(tagged_corpus):
    if (first[1] in adverbs and first[0].lower() in all_adverbs and second[1] in verbs) \
            or (second[1] in adverbs and second[0].lower() in all_adverbs and first[1] in verbs):
        # if it is an intensity modifying adverb and a verb pair, record it
        adverb_verb.append((first, second))
    elif (first[1] in adverbs and first[0].lower() in all_adverbs and second[1] in adjectives) \
            or (second[1] in adverbs and second[0].lower() in all_adverbs and first[1] in adjectives):
        # if it is an intensity modifying adverb and an adjective pair, record it
        adverb_adjective.append((first, second))

# remove duplicates of the pairs
adverb_verb = list(dict.fromkeys(adverb_verb))
adverb_adjective = list(dict.fromkeys(adverb_adjective))

print(len(all_verbs), len(all_adjectives), len(adverb_verb), len(adverb_adjective))

output = []

# for all word pairs found in the corpus that includes an intensity modifying adverb
for first, second in adverb_verb + adverb_adjective:
    if first[1] in adverbs:
        if second[1] in verbs:
            # adverb is before a verb
            output.append(('VERB', first[0], second[0], maximum_similarity(second[0], all_verb_synsets, wordnet.VERB)))
        else:
            # adverb is before an adjective
            output.append(('ADJ', first[0], second[0], maximum_similarity(second[0], all_adjective_synsets, wordnet.ADJ)))
    else:
        if first[1] in verbs:
            # adverb is after a verb
            output.append(('VERB', first[0], second[0], maximum_similarity(first[0], all_verb_synsets, wordnet.VERB)))
        else:
            # adverb is after an adjective
            output.append(('ADJ', first[0], second[0], maximum_similarity(first[0], all_adjective_synsets, wordnet.ADJ)))

# form result pairs into text
output_text = []
for out in output:
    for top in out[3][:2]:
        output_text.append(out[1] + ' ' + out[2] + ', ' + top[0] + '\n')

# save top 50 results to csv file
f = open('CS372_HW1_output_20150860.csv', 'w')
for t in output_text[:50]:
    f.write(t)
