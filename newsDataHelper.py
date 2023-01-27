import numpy

from main import allNewsData,falseDataNews, trueDataNews

def intersectNewsSubjects():
    return numpy.intersect1d(falseDataNews.subject.unique(), trueDataNews.subject.unique())