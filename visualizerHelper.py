import matplotlib.pyplot
from nltk.corpus import stopwords
import seaborn
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

simplefilter("ignore", category=ConvergenceWarning)

from main import allNewsData

def printTrueDataNews():
    matplotlib.pyplot.close()
    seaborn.countplot(allNewsData, x='category')
    matplotlib.pyplot.show()