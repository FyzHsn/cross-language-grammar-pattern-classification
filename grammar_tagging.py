import pandas as pd
import spacy

from spacy import displacy


class GrammarTagger:
    def __init__(self, df=None):
        self.df = df
        self.nlp = spacy.load("en_core_web_sm")

    def pos_tags(self, sentence):
        doc = self.nlp(sentence)
        for token in doc:
            print(token.text, token.pos_, token.tag_)
        displacy.serve(doc, style="dep")


if __name__ == "__main__":
    grammer = GrammarTagger()
    group_1 = ["The class studied",
               "The students and the teacher read.",
               "The students sat and read.",
               "The students and the teacher sat and read."]
    group_2 = ["The class took a test.",
               "The class took a test and a quiz."]
    group_3 = ["The class worked carefully.",
               "The students sit here.",
               "The class worked like a team.",
               "Before school, in the gym, the class worked like a team.",
               "In the gym, the class worked like a team before school.",
               "Like a team, the class worked before school in the gym."]
    group_4 = ["The teacher is Mr. Soto.",
               "The teachers are Mr. Soto and Ms. Lin"]
    group_5 = ["The teacher is kind.",
               "Ms. Kin is kind and helpful."]
    group_6 = ["The teacher gave the class a test",
               "Mr. Soto gave Kim and John a test."]

    # for sentence in group_6:
    #     grammer.pos_tags(sentence)
    #     print("===========")

    grammer.pos_tags("the teacher gave the class a test")