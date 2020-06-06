import nltk
import tester as tst
import re


def extract_match(np, pattern):
    np = np.flatten()
    for n in np[::-1]:
        assert type(n) == tuple
        if re.match(pattern, n[1]):
            return n[0]


def parse_with(sentence, grammar):
    cp = nltk.RegexpParser(grammar)
    return cp.parse(sentence)


class Extractor:
    def __init__(self):
        self.noun_grammar = """
        NP: {<DT>?<JJ>*<NN.?>*}
        NP: {<NP><,>?<CC><NP>}
        NP: {<NP><,><NP>}
        NP: {<CD><NP>}
        """
        self.verb_grammar = """
        VP: {<MD>*<VB.?>*<RB.?>*<VB.?><RB.?>*}
        """
        self.clause_grammar = """
        CL: {<NP><VP><NP>}
        """
        # VP: {<VB.?><NP>}
        # VP: {<VVP><NP>(<TO><VP>)?}
        # NP: {<NP><IN><VP>}
        # NP: {<NP><,>?<WDT><VP>}

    def parse_vp(self, tree):
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
        phrases = list(map(lambda p: parse_with(p, self.verb_grammar) if type(p) == list else p, phrases))
        for v in phrases:
            print('v', v, type(v))
        return phrases

    def extract(self, sentence):
        print(sentence)
        sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
        result = parse_with(sentence, self.noun_grammar)
        verbs = self.parse_vp(result)
        clauses = []
        for p in verbs:
            assert p.label() in ['S', 'NP']
            if p.label() == 'S':
                for subtree in p:
                    if type(subtree) == nltk.tree.Tree:
                        assert subtree.label() in ['S', 'VP']
                        if subtree.label() == 'S':
                            continue
                        else:
                            clauses.append(subtree)
                    else:
                        t = nltk.tree.Tree('IRR', [])
                        t.append(subtree)
                        clauses.append(t)
            else:
                clauses.append(p)
        tree = nltk.tree.Tree('S', clauses)
        clauses = parse_with(tree, self.clause_grammar)
        clauses.pretty_print()
        triples = set()
        for cl in clauses:
            if cl.label() == 'CL':
                triple = tst.Triple()
                for p in cl:
                    if p.label() == 'NP':
                        if triple.action is None:
                            triple.x = extract_match(p, 'NN.?')
                        else:
                            triple.y = extract_match(p, 'NN.?')
                    elif p.label() == 'VP':
                        triple.action = extract_match(p, 'VB.?')
                    else:
                        pass
                print(triple.to_string())
                triples.add(triple)
        return triples


tst.test_with(Extractor())
