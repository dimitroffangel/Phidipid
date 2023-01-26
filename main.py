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

stopwords = nltk.corpus.stopwords.words('english')

dataDirectoryFolder = './data/'

for dirname, _, filenames in os.walk(dataDirectoryFolder):
    for filename in filenames:
        print(os.path.join(dirname, filename))

trueDataNews = pandas.read_csv(dataDirectoryFolder + 'True.csv')
falseDataNews = pandas.read_csv(dataDirectoryFolder + 'Fake.csv')
trueDataNews['category'] = 1
falseDataNews['category'] = 0
allNewsData = pandas.concat([trueDataNews, falseDataNews])


