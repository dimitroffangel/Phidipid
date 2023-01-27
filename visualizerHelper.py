import numpy
from nltk.corpus import stopwords
import matplotlib.pyplot
import seaborn
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

simplefilter("ignore", category=ConvergenceWarning)

from main import allNewsData,falseDataNews, trueDataNews

def plotAllDataNews():
    matplotlib.pyplot.close()
    seaborn.countplot(allNewsData, x='class')
    matplotlib.pyplot.show()

## TODO::order the x label to be under the y-label 
def plotNewsSubject():
    matplotlib.pyplot.close()
    matplotlib.pyplot.figure(figsize=(16,8))
    chart=seaborn.countplot(x='subject', hue='class', data=allNewsData, linewidth=10)
    matplotlib.pyplot.show()

## TODO::order the x label to be under the y-label 
def plotFalseNewsSubject():
    matplotlib.pyplot.close()
    matplotlib.pyplot.figure(figsize=(16,8))
    chart=seaborn.countplot(x='subject',  data=falseDataNews, linewidth=10)
    matplotlib.pyplot.show()

## TODO::order the x label to be under the y-label 
def plotTrueNewsSubject():
    matplotlib.pyplot.close()
    matplotlib.pyplot.figure(figsize=(16,8))
    chart=seaborn.countplot(x='subject',  data=trueDataNews, linewidth=10)
    matplotlib.pyplot.show()