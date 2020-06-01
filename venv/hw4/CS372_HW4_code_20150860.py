import nltk
import tester as tst


class Extractor:
    def __init__(self):
        self.grammar = """
        NP: {<DT>?<JJ>*<NN.?>*}
        NP: {<NP>(<,><NP><,>)?<CC><NP>}
        NP: {<NP><IN><NP>}
        NP: {<NP><IN><VP>}
        NP: {<NP><,>?<WDT><VP>}
        VVP: {<MD>*<VB[^G.]?>*<RB.?>*<VB[^G.]?><RB.?>*}
        VP: {<VB.?><NP>}
        VP: {<VVP><NP>(<TO><VP>)?}
        CL: {<NP><VP>}
        CL: {<CL><,><CC>?<VVP><NP>}
        """

    def extract(self, sentence):
        print(sentence)
        sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
        cp = nltk.RegexpParser(self.grammar)
        result = cp.parse(sentence)
        print(result)
        # result.draw()
        return {}


tst.test_with(Extractor())
