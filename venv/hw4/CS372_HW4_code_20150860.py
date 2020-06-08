import nltk
import tester as tst
import re
from nltk.stem.snowball import SnowballStemmer


'''extract noun from a noun phrase tree'''
def extract_noun(phrase):
    if type(phrase) == nltk.tree.Tree:
        words = list(map(lambda p: p[0], phrase.flatten()))
        # maybe with and in should precede and/or??
        if 'and' in words or 'or' in words:
            # take the whole tree
            phrase = list(map(lambda p: p[0], phrase.flatten()))
            result = ' '.join(phrase)
            result = re.sub("\s+,", ",", result)
        elif 'of' in words or 'by' in words or 'in' in words or 'about' in words or 'for' in words or 'from' in words or 'under' in words or 'with' in words or 'than' in words:
            # take the first subtree
            result = extract_noun(phrase[0])
        else:
            phrase = list(map(lambda p: p[0], phrase.flatten()))
            result = ' '.join(phrase)
            result = re.sub("\s+,", ",", result)
        result = nltk.word_tokenize(result)
        # remove articles(DT)
        if result[0].lower() in ['the', 'a', 'an']:
            result = result[1:]
        result = ' '.join(result)
        result = re.sub("\s+,", ",", result)
        return result
    else:
        return phrase[0]


'''extract verbs from the provided verb phrase'''
def extract_verb(phrase):
    phrase = phrase.flatten()
    phrase = list(filter(lambda p: not re.match('(RB.?)|(JJ.?)', p[1]) if p[0] not in ['not'] else True, phrase))
    phrase = list(map(lambda p: p[0], phrase))
    return ' '.join(phrase)


'''parse the given sentence with the provided grammar rules'''
def parse_with(sentence, grammar):
    cp = nltk.RegexpParser(grammar)
    return cp.parse(sentence)


'''determine of the provided word is an inflection of one of verbs in verbs list.'''
def is_inflection_of(verbs, words):
    verbs = list(map(lambda w: stemmer.stem(w), verbs))
    verbs_regex = "^(" + ")|^(".join(verbs) + ")"
    # ignore gerund forms
    words = list(filter(lambda w: not w.endswith('ing'), nltk.word_tokenize(words)))
    word_match = list(map(lambda w: re.match(verbs_regex, w), words))
    return any(word_match)


relevant_verbs = ['activate', 'inhibit', 'bind']
positive_verbs = ['accelerate', 'augment', 'induce', 'stimulate', 'require', 'up-regulate']
negative_verbs = ['abolish', 'block', 'down-regulate', 'prevent']

stemmer = SnowballStemmer("english")

'''
Class definition for the Extractor class.
Encapsulates grammar rules, parsing functions and extracting functions.
'''
class Extractor:
    def __init__(self):
        self.noun_grammar = """
        NP: {<DT>?<JJ.?>*<NN.?>+}
        NP: {<PRP.?><NN.?>+}
        NP: {<NP><,>?<CC><NP>}
        NP: {<NP><,><NP>}
        NP: {<CD><NP>}
        NP: {<NP><IN><NP>}
        """
        self.active_verb_grammar = """
        ACT: {<MD>*<VB.?>*<RB.?>*<VB.?><RB.?>*}
        """
        self.passive_verb_grammar = """
        PSV: {<VBD><RB.?>*<VBN><RB.?>*<IN>}
        """
        self.clause_grammar = """
        CL: {<NP><ACT|PSV><NP>}
        """
        # VP: {<VB.?><NP>}
        # VP: {<VVP><NP>(<TO><VP>)?}
        # NP: {<NP><IN><VP>}
        # NP: {<NP><,>?<WDT><VP>}

    '''parses passive forms of verbs from a given tree of a sentence.'''
    def parse_passive_verbs(self, tree):
        phrases = []
        phrase = []
        # only consider words that are not part of other phrases (e.g. NP)
        for s in tree:
            if type(s) == tuple:
                phrase.append(s)
            else:
                if len(phrase) > 0:
                    phrases.append(phrase)
                    phrase = []
                phrases.append(s)
        phrases_parsed = []
        # parse lists of words that are not part of other phrases.
        for p in phrases:
            if type(p) == list:
                parsed = parse_with(p, self.passive_verb_grammar)
                for ps in parsed:
                    phrases_parsed.append(ps)
            else:
                phrases_parsed.append(p)
        return phrases_parsed

    '''parses active forms of verbs from a given tree of a sentence.'''
    def parse_active_verbs(self, tree):
        phrases = []
        phrase = []
        for s in tree:
            if type(s) == tuple:
                phrase.append(s)
            else:
                if len(phrase) > 0:
                    phrases.append(phrase)
                    phrase = []
                phrases.append(s)
        # parse lists of words that are not part of other phrases.
        phrases = list(map(lambda p: parse_with(p, self.active_verb_grammar) if type(p) == list else p, phrases))
        return phrases

    def extract(self, sentence):
        print(sentence)
        sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
        # parse noun phrases first
        result = parse_with(sentence, self.noun_grammar)
        # then parse passive forms of verbs
        result = self.parse_passive_verbs(result)
        # then parse active forms of verbs
        result = self.parse_active_verbs(result)
        clauses = []
        # merge trees of NP, VP and other nodes into a single tree
        for p in result:
            assert p.label() in ['S', 'NP', 'ACT', 'PSV']
            if p.label() == 'S':
                for subtree in p:
                    if type(subtree) == nltk.tree.Tree:
                        assert subtree.label() in ['S', 'ACT', 'PSV']
                        if subtree.label() == 'S':
                            continue
                        else:
                            clauses.append(subtree)
                    else:
                        # irrelevant nodes
                        t = nltk.tree.Tree('IRR', [])
                        t.append(subtree)
                        clauses.append(t)
            else:
                clauses.append(p)
        tree = nltk.tree.Tree('S', clauses)
        clauses = parse_with(tree, self.clause_grammar)
        clauses.pretty_print()
        triples = set()
        # extract triples from clauses which contain NP and VP
        for cl in clauses:
            if cl.label() == 'CL':
                triple = tst.Triple()
                for p in cl:
                    if p.label() == 'NP':
                        if triple.action is None:
                            triple.x = extract_noun(p)
                        else:
                            triple.y = extract_noun(p)
                    elif p.label() in ['ACT', 'PSV']:
                        triple.action = extract_verb(p)
                    else:
                        pass
                triples.add(triple)
        # filter out triples that are not relevant verbs
        triples = set(filter(lambda t: is_inflection_of(relevant_verbs, t.action)
                                       or is_inflection_of(positive_verbs, t.action)
                                       or is_inflection_of(negative_verbs, t.action), triples))
        print(triples)
        return triples


tst.test_with(Extractor())
