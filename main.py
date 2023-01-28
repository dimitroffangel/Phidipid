import itertools
import numpy
import nltk
import matplotlib.pyplot
from nltk.corpus import stopwords
import os
import pandas
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from warnings import simplefilter

simplefilter("ignore", category=ConvergenceWarning)

nltk.download('stopwords')
hermesDataDirectoryFolder = './data/hermes/'
mercuryDataDirectoryFolder = './data/mercury/'

stopwords = nltk.corpus.stopwords.words('english')

trueDataNews = pandas.read_csv(hermesDataDirectoryFolder + 'True.csv')
falseDataNews = pandas.read_csv(hermesDataDirectoryFolder + 'Fake.csv')
falseMercuryDataNews = pandas.read_csv(mercuryDataDirectoryFolder + 'fake.csv')
englishMercuryDataNews = falseMercuryDataNews.loc[(falseMercuryDataNews['language'] == 'english') & (falseMercuryDataNews['title'].notna())]

trueDataNews['class'] = 1
falseDataNews['class'] = 0
allNewsData = pandas.concat([trueDataNews, falseDataNews])