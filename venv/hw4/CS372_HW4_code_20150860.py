import nltk
import tester as tst


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
        phrases = list(map(lambda p: self.parse_with(p, self.verb_grammar) if type(p) == list else p, phrases))
        for v in phrases:
            print('v', v, type(v))
        return phrases

    def parse_with(self, sentence, grammar):
        cp = nltk.RegexpParser(grammar)
        return cp.parse(sentence)

    def extract(self, sentence):
        print(sentence)
        sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
        result = self.parse_with(sentence, self.noun_grammar)
        verbs = self.parse_vp(result)
        clauses = []
        for p in verbs:
            assert p.label() in ['S', 'NP']
            if p.label() == 'S':
                for subtree in p:
                    if type(subtree) == nltk.tree.Tree:
                        subtree.pretty_print()
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
        tree.pretty_print()
        return {}


tst.test_with(Extractor())
