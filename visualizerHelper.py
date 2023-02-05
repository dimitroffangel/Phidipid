import itertools
import numpy
from nltk.corpus import stopwords
import matplotlib.pyplot
import seaborn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
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
    chart=seaborn.countplot(x='subject',  data=falseDataNews, linewidth=10)
    matplotlib.pyplot.show()

def plotData(data):
    matplotlib.pyplot.close()
    # barplot=seaborn.barplot(data=data, x='class', y='numberOfTweets', estimator='scalar')
    data.plot.bar(x='class', y='numberOfTweets', rot = 0)
    matplotlib.pyplot.show()

## TODO::order the x label to be under the y-label 
def plotTrueNewsSubject():
    matplotlib.pyplot.close()
    chart=seaborn.countplot(x='subject',  data=trueDataNews)
    matplotlib.pyplot.show()

def plot_confusion_matrix(correctTargetValues, estimatedTargetValues, classes, windowTitle='Figure', bNormalizeConfusionMatrix=False, title='Confusion Matrix', colourMap=matplotlib.pyplot.cm.Blues, figureSize=(9,7)):
    matrix = confusion_matrix(correctTargetValues, estimatedTargetValues)

    if bNormalizeConfusionMatrix:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, numpy.newaxis]
    
    matplotlib.pyplot.close()
    matplotlib.pyplot.figure(num=windowTitle)
    matplotlib.pyplot.imshow(matrix, interpolation='nearest', cmap=colourMap)
    matplotlib.pyplot.title(title)

    tickMarks = numpy.arange(len(classes))
    matplotlib.pyplot.xticks(tickMarks, classes, rotation=45)
    matplotlib.pyplot.yticks(tickMarks, classes)

    fmt= '.2f' if bNormalizeConfusionMatrix else 'd'
    threshold = matrix.max() / 2.0
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        matplotlib.pyplot.text(j, i, format(matrix[i,j], fmt),
            horizontalalignment='center',
            size=int((figureSize[0] / 10) * 38),
            color='white' if matrix[i,j] > threshold else 'black')

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.ylabel('True label')
    matplotlib.pyplot.xlabel('Predicted lable')
    matplotlib.pyplot.show()
