import math
import numpy as np
import pandas as pd
import spacy

from collections import namedtuple
from spacy import displacy

SentenceData = namedtuple('SentenceData', 'sentence_id language sentence ' 
                                          'token_freq')


class GrammarTagger:
    def __init__(self, df=None):
        self.df = df
        self.data = None
        self.token_doc_freq = {}
        self.token_id = {}
        self.inverse_token_id = {}
        self.token_num = 0
        self.nlp = spacy.load("en_core_web_sm")

    def _add_new_token(self, token):
        self.token_num += 1
        self.token_id[token] = (self.token_num - 1)
        self.inverse_token_id[(self.token_num - 1)] = token

    def _update_vocabulary(self, tokens):
        for token in tokens:
            if token in self.token_doc_freq:
                self.token_doc_freq[token] += 1
            else:
                self._add_new_token(token)
                self.token_doc_freq[token] = 1

    def _get_sentence_tokens(self, sentence):
        sentence = self.nlp(sentence)
        sentence_token_freq = {}
        for token in sentence:
            for tag in [token.tag_, token.dep_]:
                if tag in sentence_token_freq:
                    sentence_token_freq[tag] += 1
                else:
                    sentence_token_freq[tag] = 1
        return sentence_token_freq

    def generate_tag_data(self):
        self.data = []
        for index, row in self.df.iterrows():
            token_freq = self._get_sentence_tokens(row['sentence'])
            self._update_vocabulary(list(token_freq.keys()))
            self.data.append(SentenceData(sentence_id=row['sentence_id'],
                                          language=row['language'],
                                          sentence=row['sentence'],
                                          token_freq=token_freq))

    def _tfidf(self, term_frequency, document_frequency):
        doc_num = self.df.shape[0]
        inverse_document_frequency = \
            math.log((1.0 + doc_num) / (1.0 + document_frequency)) + 1
        return term_frequency * inverse_document_frequency

    def tfidf_transform(self):
        tfidf_feat = np.zeros(shape=(self.df.shape[0], self.token_num))
        for row_idx, row in enumerate(self.data):
            for token, term_freq in row.token_freq.items():
                col_idx = self.token_id[token]
                doc_freq = self.token_doc_freq[token]
                tfidf_feat[row_idx][col_idx] = self._tfidf(term_freq, doc_freq)
        return tfidf_feat




if __name__ == "__main__":
    # df = pd.read_csv('data/raw_data.csv')
    # df_subset = df[df['language'] == 'eng'].copy()
    # grammar = GrammarTagger(df_subset)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The councimans daughter applied for a job with the city.")
    displacy.serve(doc, style="dep")