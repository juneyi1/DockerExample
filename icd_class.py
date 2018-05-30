import pandas as pd
from string import digits, punctuation
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

class Tokenization(object):
    """tokenize each word and for each tocken:
    1. excludes digits and punctuations, 
    2. excludes token that has '-year', 
    3. excludes stop words,
    4. makes the token stemmed,
    and return a string with all tokens separated by a space"""
    # from string import digits, punctuation
    # from nltk.tokenize import word_tokenize
    # from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    # from nltk.stem import PorterStemmer

    def __init__(self, #nlp=word_tokenize()#spacy.load('en_core_web_sm'), 
                 stop=list(ENGLISH_STOP_WORDS)+['year', 'month', 'old'], 
                 stemmer=PorterStemmer()):
        #self.nlp = nlp
        self.stop = stop
        self.stemmer = stemmer

    def fit(self, x, y=None):
        return self
        
    def transform(self, x):
        def words2tokens(sentence, stop=self.stop, stemmer=self.stemmer):
            l = []    
            tokens = word_tokenize(sentence)
            for token in tokens:                
                t = token.lower()
                if (t in punctuation) or t.isdigit():
                    continue
                if ('-year' in t) or ('-month' in t) or ('-day' in t):
                    continue
                if ('/' in t) or ('=' in t) or (t in stop):
                    continue
                l.append(stemmer.stem(t))
            return ' '.join(l)

        series = x.apply(lambda s: words2tokens(s, stop=self.stop, stemmer=self.stemmer))    
        return series
        
class MatrixConverter(object):
    """Converter a tfidf matrix to a df"""
    def __init__(self):
        pass
    def fit(self, x, y=None):
        return self    
    def transform(self, x):
        return pd.DataFrame(x.todense())