import numpy
import re as regularExpression
from nltk.stem import PorterStemmer

from main import falseDataNews, trueDataNews, stopwords

def intersectNewsSubjects():
    return numpy.intersect1d(falseDataNews.subject.unique(), trueDataNews.subject.unique())

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