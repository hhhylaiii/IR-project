# http://tartarus.org/~martin/PorterStemmer/python.txt
from PorterStemmer import PorterStemmer

class Parser:
    #A processor for removing the commoner morphological and inflexional endings from words in English
    stemmer=None
    stopwords=[]

    def __init__(self,):
        self.stemmer = PorterStemmer()
        # English stopwords from ftp://ftp.cs.cornell.edu/pub/smart/english.stop
        self.stopwords = set(open('english.stop', 'r').read().split())
        self._stem_cache = {}

    def clean(self, string):
        """ remove any nasty grammar tokens from string """
        string = string.replace(".","")
        string = string.replace(r"\s+"," ")
        string = string.lower()
        return string

    def removeStopWords(self, list):
        """ Remove common words which have no search value """
        return [word for word in list if word not in self.stopwords ]

    def tokenise(self, string):
        """ break string up into tokens and stem words """
        string = self.clean(string)
        words = string.split(" ")

        stem = self.stemmer.stem
        cache = self._stem_cache
        out = []
        for w in words:
            if not w:
                continue
            s = cache.get(w)
            if s is None:
                s = stem(w, 0, len(w)-1)
                cache[w] = s
            out.append(s)
        return out
