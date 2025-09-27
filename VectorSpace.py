from pprint import pprint
from Parser import Parser
import util
from math import log

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents=[], doc_names=[]):
        self.documentVectors=[]
        self.parser = Parser()
        self.doc_names = doc_names
        if(len(documents)>0):
            self.build(documents)

    def load_documents(self, folder_path):
        import os, re
        self.doc_names = []
        self.documents = []
        for filename in sorted(os.listdir(folder_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
            if(filename.endswith(".txt")):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    self.documents.append(file.read())
                    self.doc_names.append(filename)
        return self.doc_names, self.documents

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents] #TF weighted document vectors

        df_count = [0] * len(self.vectorKeywordIndex)
        for doc in documents:
            wordList = self.parser.tokenise(doc)
            wordList = self.parser.removeStopWords(wordList)
            seen = set()
            for word in wordList:
                idx = self.vectorKeywordIndex.get(word)
                if idx is not None and idx not in seen: # Check if the index is not None and not seen before
                    df_count[idx] += 1
                    seen.add(idx)
        N = len(documents)
        self.idf = [log(N / df_count[i]) if df_count[i] > 0 else 0.0 for i in range(len(df_count))]

        self.documentVectors_tfidf = [] # TF-IDF weighted document vectors
        for tf in self.documentVectors:
            tfidf = [ tf[i] * self.idf[i] for i in range(len(self.vectorKeywordIndex)) ]
            self.documentVectors_tfidf.append(tfidf)

        #print(self.vectorKeywordIndex)
        #print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        vectorTF = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vectorTF[self.vectorKeywordIndex[word]] += 1; #Use TF Model
        return vectorTF

    def buildQueryVector(self, termList, use_tfidf=False):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        if use_tfidf:
            return [query[i] * self.idf[i] for i in range(len(query))]
        return query


    #def related(self,documentId):
        #""" find documents that are related to the document indexed by passed Id within the document Vectors"""
        #ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        #return ratings


    def search(self, searchList, use_tfidf=False):
        """ search for documents that match based on a list of terms """
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
        resultswitheuclidean_d.sort(key=lambda x: x[1], reverse=False)  # For Euclidean distance, lower is better
        return resultswithcosine[:10], resultswitheuclidean_s[:10], resultswitheuclidean_d[:10]  # Return top 10 results


if __name__ == '__main__':

    vectorSpace = VectorSpace()

    vectorSpace.load_documents("EnglishNews")

    vectorSpace = VectorSpace(vectorSpace.documents, vectorSpace.doc_names)

    #print(vectorSpace.doc_names[:5], vectorSpace.documents[:5])

    #print(vectorSpace.vectorKeywordIndex)

    #print(vectorSpace.documentVectors)

    #print(vectorSpace.related(1))

    resultswithcosineTF, resultswitheuclidean_s_TF , resultswitheuclidean_d_TF = vectorSpace.search(["planet Taiwan typhoon"], use_tfidf=False) #TF weighted search
    resultswithcosineTFIDF, resultswitheuclidean_s_TFIDF, resultswitheuclidean_d_TFIDF = vectorSpace.search(["planet Taiwan typhoon"], use_tfidf=True) #TF-IDF weighted search

    #TF Cosine similarity
    print("TF Cosine")
    print(f"{'NewsID':<15}{'Score':>6}")
    for name, score in resultswithcosineTF:
        print(f"{name:<15}{score:>10.7f}")

    print("-"*40)

    #TF-IDF Cosine similarity
    print("TF-IDF Cosine")
    print(f"{'NewsID':<15}{'Score':>6}")
    for name, score in resultswithcosineTFIDF:
        print(f"{name:<15}{score:>10.7f}")

    print("-"*40)

    #TF Euclidean similarity
    print("TF Euclidean")
    print(f"{'NewsID':<15}{'Score':>6}")
    for name, score in resultswitheuclidean_s_TF:
        print(f"{name:<15}{score:>10.7f}")

    print("-"*40)

    #TF-IDF Euclidean distance
    print("TF-IDF Euclidean")
    print(f"{'NewsID':<15}{'Score':>6}")
    for name, score in resultswitheuclidean_d_TFIDF:
        print(f"{name:<15}{score:>10.7f}")

    print("-"*40)

###################################################
