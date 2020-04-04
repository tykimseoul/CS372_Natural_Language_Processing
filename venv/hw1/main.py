import nltk
from nltk.corpus import brown, wordnet
import itertools
from math import floor
from nltk.stem.snowball import SnowballStemmer


def all_occurrences(pos, corpus):
    occurrences = list(filter(lambda a: a[1] in pos, corpus))
    occurrences = list(map(lambda a: a[0], occurrences))
    return list(dict.fromkeys(occurrences))


def adverbs_of_degree():
    seeds = ["extremely", "quite", "just", "almost", "very", "too", "enough", "slightly", "completely"]
    generated = list(map(lambda s: wordnet.synsets(s, pos=wordnet.ADV), seeds))
    generated = [item.lemma_names()[0] for sublist in generated for item in sublist]
    generated = set(generated)
    generated = set(filter(lambda g: g in raw_corpus, generated))
    print(generated)
    return generated


stemmer = SnowballStemmer("english")
tagged_corpus = brown.tagged_words(categories="news")
raw_corpus = brown.words()
print("corpus length", len(tagged_corpus))
verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjectives = ['JJ', 'JJR', 'JJS']
adverbs = ['RB', "RBR", "RBS"]
all_adverbs = adverbs_of_degree()
all_verbs = all_occurrences(verbs, tagged_corpus)
all_adjectives = all_occurrences(adjectives, tagged_corpus)
all_verb_synsets = dict(map(lambda s: (s, wordnet.synsets(s, pos=wordnet.VERB)), all_verbs))
all_adjective_synsets = dict(map(lambda s: (s, wordnet.synsets(s, pos=wordnet.ADJ)), all_adjectives))
adverb_verb = []
adverb_adjective = []


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def maximum_similarity(word, target_synsets, pos):
    word_synset = wordnet.synsets(word, pos=pos)
    word_stem = stemmer.stem(word)
    similarities = []
    for target in target_synsets:
        target_stem = stemmer.stem(target)
        if word_stem != target_stem:
            sim = synset_similarity(word_synset, target_synsets[target])
            similarities.append((target, sim, stemmer.stem(target)))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:50]
    stems = set(map(lambda t: t[2], similarities))
    filtered = []
    for sim in similarities:
        if len(stems) == 0:
            break
        if sim[2] in stems:
            filtered.append((sim[0], sim[1]))
            # filtered.append(sim)
            stems.remove(sim[2])
    return filtered[:5]


def synset_similarity(first_set, second_set):
    limit = 0.3
    if len(first_set) == 0 or len(second_set) == 0:
        return 0
    scores = []
    for f in first_set[:max(floor(len(first_set) * limit), 1)]:
        for s in second_set[:max(floor(len(second_set) * limit), 1)]:
            score = safe_similarity(f, s)
            scores.append(score)
    return sum(scores) / len(scores)


def safe_similarity(f, s):
    sim1 = f.path_similarity(s)
    sim2 = s.path_similarity(f)
    if sim1 != sim2:
        if sim1 is None:
            return sim2
        else:
            return sim1
    else:
        if sim1 is None:
            return 0
        else:
            return sim1


for first, second in pairwise(tagged_corpus):
    if (first[1] in adverbs and first[0] in all_adverbs and second[1] in verbs) \
            or (second[1] in adverbs and second[0] in all_adverbs and first[1] in verbs):
        adverb_verb.append((first, second))
    if (first[1] in adverbs and first[0] in all_adverbs and second[1] in adjectives) \
            or (second[1] in adverbs and second[0] in all_adverbs and first[1] in adjectives):
        adverb_adjective.append((first, second))

# remove duplicates
adverb_verb = list(dict.fromkeys(adverb_verb))
adverb_adjective = list(dict.fromkeys(adverb_adjective))

print(len(all_verbs), len(all_adjectives), len(adverb_verb), len(adverb_adjective))

for first, second in adverb_verb + adverb_adjective:
    if first[1] in adverbs:
        if second[1] in verbs:
            print("VERB: ", first[0], second[0], maximum_similarity(second[0], all_verb_synsets, wordnet.VERB))
        else:
            print("ADJ: ", first[0], second[0], maximum_similarity(second[0], all_adjective_synsets, wordnet.ADJ))
    else:
        if first[1] in verbs:
            print("VERB: ", first[0], second[0], maximum_similarity(first[0], all_verb_synsets, wordnet.VERB))
        else:
            print("ADJ: ", first[0], second[0], maximum_similarity(first[0], all_adjective_synsets, wordnet.ADJ))
