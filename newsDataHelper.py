import numpy
import re as regularExpression
from nltk.stem import PorterStemmer

from main import stopwords

def intersectNewsSubjects(lhsData, rhsData):
    return numpy.intersect1d(lhsData.subject.unique(), rhsData.subject.unique())

def removeNoisyData(data, numberOfWords):
    corpus = []
    for i in range(0, numberOfWords):
        document = regularExpression.sub('[^a-zA-Z]', ' ', data[i])
        document = document.lower()
        document = document.split()
        porterStemmer = PorterStemmer()
        document = [porterStemmer.stem(word) for word in document if not word in set(stopwords)]
        document = ' '.join(document) 
        corpus.append(document)
    return corpus

def countUniqueWords(listOfDocuments):
    foundWords = set()
    for document in listOfDocuments:
        words = document.split()
        for word in words:
            if word not in foundWords:
                foundWords.add(word)
    return foundWords
