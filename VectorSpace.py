from pprint import pprint
from Parser import Parser
import util

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
        self.documentVectors = [self.makeVector(document) for document in documents]

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

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use TF Model
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    #def related(self,documentId):
        #""" find documents that are related to the document indexed by passed Id within the document Vectors"""
        #ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        #return ratings


    def search(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratingswithcosine = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        ratingswitheuclidean = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]

        resultswithcosine = list(zip(self.doc_names, ratingswithcosine))
        resultswitheuclidean = list(zip(self.doc_names, ratingswitheuclidean))

        resultswithcosine.sort(key=lambda x: x[1], reverse=True)
        resultswitheuclidean.sort(key=lambda x: x[1], reverse=True)
        return resultswithcosine[:10], resultswitheuclidean[:10]  # Return top 10 results


if __name__ == '__main__':

    vectorSpace = VectorSpace()

    vectorSpace.load_documents("EnglishNews")

    vectorSpace = VectorSpace(vectorSpace.documents, vectorSpace.doc_names)

    #print(vectorSpace.doc_names[:5], vectorSpace.documents[:5])

    #print(vectorSpace.vectorKeywordIndex)

    #print(vectorSpace.documentVectors)

    #print(vectorSpace.related(1))

    resultswithcosine, resultswitheuclidean = vectorSpace.search(["planet Taiwan typhoon"])

    #TF Cosine
    print("TF Cosine")
    print(f"{'NewsID':<15}{'Score':>6}")
    for name, score in resultswithcosine:
        print(f"{name:<15}{score:>10.7f}")

    print("-"*40)

    #TF Euclidean
    print("TF Euclidean")
    print(f"{'NewsID':<15}{'Score':>6}")
    for name, score in resultswitheuclidean:
        print(f"{name:<15}{score:>10.7f}")

###################################################
