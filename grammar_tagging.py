import pandas as pd
import spacy

from sklearn.feature_extraction.text import \
    CountVectorizer, \
    TfidfVectorizer
from spacy import displacy


class GrammarTagger:
    def __init__(self, df=None):
        self.df = df
        self.nlp = spacy.load("en_core_web_sm")
        self.X = None

    def generate_tags(self):
        tags = []
        for index, row in self.df.iterrows():
            doc = self.nlp(row['sentence'])
            for token in doc:
                tags += [token.tag_, token.dep_]
        self.df['tags'] = " ".join(tags)

    def get_vectorize_data(self):
        vectorizer = CountVectorizer()
        corpus = self.df['tags'].values
        X = vectorizer.fit_transform(corpus)
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(X)
        return X


if __name__ == "__main__":
    df = pd.read_csv('data/raw_data.csv')
    df_subset = df[df['language'] == 'eng'].copy()
    grammar = GrammarTagger(df_subset)
    grammar.generate_tags()
    print(grammar.df.head())
    print(grammar.df.tail())
    print(grammar.get_vectorize_data())
