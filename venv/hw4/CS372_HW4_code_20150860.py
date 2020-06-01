import nltk
import tester as tst


class Extractor:
    def __init__(self):
        self.grammar = """NP: {(<DT>?<JJ>*<NN.?>*)}
                       VP: {<MD>*<VB.?>*<RB.?>*<VB.?><RB.?>*<IN>?}"""

    def extract(self, sentence):
        sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
        cp = nltk.RegexpParser(self.grammar)
        result = cp.parse(sentence)
        print(result)
        # result.draw()
        return {}


tst.test_with(Extractor())
