import itertools
import numpy
import nltk
import matplotlib.pyplot
from nltk.corpus import stopwords
import os
import pandas
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from warnings import simplefilter

import main
import visualizerHelper
import newsDataHelper

simplefilter("ignore", category=ConvergenceWarning)

numberOfWords = 42000
corpus = newsDataHelper.removeNoisyData(main.allNewsData['title'].values, numberOfWords)

countVectorizer = CountVectorizer(max_features=10000)
X = countVectorizer.fit_transform(corpus).toarray()
y = main.allNewsData['class'].values[:numberOfWords]
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=0)
classifier = MultinomialNB(alpha=0.01)
crossValueScoreResult = cross_val_score(classifier, xTrain, yTrain, cv=3, n_jobs=6)
predictionTrainY = cross_val_predict(classifier, xTrain, yTrain, cv=3, n_jobs=6)
classifier.fit(xTrain, yTrain)
predictTestY = classifier.predict(xTest)

def plotConfusionMatrix(yExpected, yPredicted, classes):
    print(accuracy_score(yExpected, yPredicted))
    visualizerHelper.plot_confusion_matrix(yExpected, yPredicted, classes=classes)
    visualizerHelper.plot_confusion_matrix(yExpected, yPredicted, classes=classes, bNormalizeConfusionMatrix=True)

plotConfusionMatrix(yTrain, predictionTrainY, main.allNewsData['class'].unique())
plotConfusionMatrix(yTest, predictTestY, main.allNewsData['class'].unique())
