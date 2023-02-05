from nltk.stem import PorterStemmer
import numpy
import re as regularExpression
from sklearn.feature_extraction.text import CountVectorizer

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

def wordsFrequency(listOfDocuments):
    countVectorizer = CountVectorizer().fit(listOfDocuments)
    transformedDocuments = countVectorizer.transform(listOfDocuments)
    continousBagOfWords = transformedDocuments.sum(axis= 0)
    print(continousBagOfWords)
    print(countVectorizer.vocabulary_.items())
    wordDictioanry = [(currentWord, continousBagOfWords[0, currentWordIndex]) for currentWord, currentWordIndex in countVectorizer.vocabulary_.items()]
    sortedWordDictionary = sorted(wordDictioanry, key = lambda x: x[1], reverse=True)
    return sortedWordDictionary
