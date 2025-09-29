from Parser import Parser
from ParserZH import ParserZH
import util
from math import log
import numpy as np
import sys

class VectorSpace:
    """A algebraic model for representing text documents as vectors of identifiers."""

    documentVectors = []
    vectorKeywordIndex = []
    parser = None

    def __init__(self, documents=[], doc_names=[]):
        self.documentVectors = []
        self.parser = Parser()
        self.doc_names = doc_names
        if len(documents) > 0:
            self.build(documents)

    def load_documents(self, folder_path):
        import os, re

        def _read_text(path):
            for enc in ('utf-8', 'utf-8-sig', 'cp950', 'big5hkscs', 'big5', 'latin-1'):
                try:
                    with open(path, 'r', encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        self.doc_names = []
        self.documents = []
        for filename in sorted(os.listdir(folder_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
            if filename.endswith(".txt"):
                full = os.path.join(folder_path, filename)
                self.documents.append(_read_text(full))
                self.doc_names.append(filename)
        return self.doc_names, self.documents

    def build(self, documents):
        self.documents = documents
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)

        V = len(self.vectorKeywordIndex)
        self._df_count = [0] * V
        self._accumulating_df = True

        self.documentVectors = [self.makeVector(document) for document in documents]

        self._accumulating_df = False

        N = len(documents)
        self.idf = [log(N / self._df_count[i]) if self._df_count[i] > 0 else 0.0 for i in range(V)]
        self._idf_np = np.array(self.idf, dtype=np.float32)

        self.documentVectors = [np.array(tf, dtype=np.float32) for tf in self.documentVectors]
        self.documentVectors_tfidf = [tf * self._idf_np for tf in self.documentVectors]

    def getVectorKeywordIndex(self, documentList):
        vocabularyString = " ".join(documentList)
        vocabularyList = self.parser.tokenise(vocabularyString)
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex = {}
        offset = 0
        for word in uniqueVocabularyList:
            vectorIndex[word] = offset
            offset += 1
        return vectorIndex

    def makeVector(self, wordString):
        V = len(self.vectorKeywordIndex)
        vectorTF = [0] * V

        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)

        seen = set() if getattr(self, "_accumulating_df", False) else None
        vki = self.vectorKeywordIndex

        for word in wordList:
            idx = vki.get(word)
            if idx is None:
                continue
            if seen is not None and vectorTF[idx] == 0:
                seen.add(idx)
            vectorTF[idx] += 1

        if seen is not None:
            for idx in seen:
                self._df_count[idx] += 1

        return vectorTF

    def buildQueryVector(self, termList, use_tfidf=False):
        V = len(self.vectorKeywordIndex)
        q = np.zeros(V, dtype=np.float32)
        wordList = self.parser.tokenise(" ".join(termList))
        wordList = self.parser.removeStopWords(wordList)
        vki = self.vectorKeywordIndex
        for w in wordList:
            idx = vki.get(w)
            if idx is not None:
                q[idx] += 1.0
        if use_tfidf:
            return q * self._idf_np
        return q

    def search(self, searchList, use_tfidf=False):
        docModel = self.documentVectors_tfidf if use_tfidf else self.documentVectors
        queryVector = self.buildQueryVector(searchList, use_tfidf=use_tfidf)

        ratingswithcosine = [util.cosine(queryVector, documentVector) for documentVector in docModel]
        ratingswitheuclidean_s = [util.euclideanSimilarity(queryVector, documentVector) for documentVector in docModel]
        ratingswitheuclidean_d = [util.euclidean(queryVector, documentVector) for documentVector in docModel]

        resultswithcosine = list(zip(self.doc_names, ratingswithcosine))
        resultswitheuclidean_s = list(zip(self.doc_names, ratingswitheuclidean_s))
        resultswitheuclidean_d = list(zip(self.doc_names, ratingswitheuclidean_d))

        resultswithcosine.sort(key=lambda x: x[1], reverse=True)
        resultswitheuclidean_s.sort(key=lambda x: x[1], reverse=True)
        resultswitheuclidean_d.sort(key=lambda x: x[1], reverse=False)
        return resultswithcosine[:10], resultswitheuclidean_s[:10], resultswitheuclidean_d[:10]

    def extract_nv_terms(self, doc):
        try:
            import nltk
            from nltk import pos_tag, word_tokenize
        except Exception:
            print("Error: Requires nltk from https://www.nltk.org/. Have you installed it?")
            sys.exit(1)

        tokens = [t for t in word_tokenize(doc) if t.isalpha()]
        tagged = pos_tag(tokens)
        extracted_doc = [term.lower() for (term, tag) in tagged if tag.startswith('NN') or tag.startswith('VB')]
        return self.parser.tokenise(" ".join(extracted_doc))

    def feedback_research(self, original_query, x=1.0, y=0.5):
        original_query_results, _, _ = self.search(original_query, use_tfidf=True)
        top1_name = original_query_results[0][0]
        top1_index = self.doc_names.index(top1_name)
        nv_terms = self.extract_nv_terms(self.documents[top1_index])

        q_orig = self.buildQueryVector(original_query, use_tfidf=True)
        q_fb = self.buildQueryVector(nv_terms, use_tfidf=True)
        q_new = x * q_orig + y * q_fb

        ratingswithcosine = [util.cosine(q_new, documentVector) for documentVector in self.documentVectors_tfidf]
        resultswithcosine = list(zip(self.doc_names, ratingswithcosine))
        resultswithcosine.sort(key=lambda x: x[1], reverse=True)
        return resultswithcosine[:10]
