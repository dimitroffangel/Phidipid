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

dataDirectoryFolder = '.\data'

for dirname, _, filenames in os.walk(dataDirectoryFolder):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pandas.read_csv(dataDirectoryFolder + '/train.csv', index_col='id')
test = pandas.read_csv(dataDirectoryFolder + '/test.csv', index_col='id')

vectorizer = CountVectorizer()


documents = numpy.array(['Mirabai has won a silver medal in weight lifting in Tokyo olympics 2021',
                 'Sindhu has won a bronze medal in badminton in Tokyo olympics',
                 'Indian hockey team is in top four team in Tokyo olympics 2021 after 40 years'])

wordsInTrainTexts = train['tweet'].str.split(expand=True).unstack().value_counts()
bag = vectorizer.fit_transform(train)
pipeline = Pipeline([
    ('features', CountVectorizer()),
    ('clf', LinearSVC())
])
crossValueScoreResult = cross_val_score(pipeline, train.tweet, train.label, cv=10, n_jobs=6)
res=pipeline.fit(train.tweet, train.label)
prediction = cross_val_predict(pipeline, train.tweet, train.label, cv=3, n_jobs=6)

counter = 0
a = train.label.values
print(a[0])
for i in range(len(prediction)):
    if prediction[i] != a[i]:
        print(train.tweet.values[i])



def plot_confusion_matrix(correctTargetValues, estimatedTargetValues, classes, bNormalizeConfusionMatrix=False, title='Confusion Matrix', colourMap=matplotlib.pyplot.cm.Blues, figureSize=(9,7)):
    matrix = confusion_matrix(correctTargetValues, estimatedTargetValues)

    if bNormalizeConfusionMatrix:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, numpy.newaxis]
    
    matplotlib.pyplot.close()
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

labels = pipeline.classes_
# plot_confusion_matrix(train.label, prediction, classes=labels)
test_results = pipeline.predict(test.tweet.values)

# for i in range(len(test_results)):
#     if test_results[i]:
#         print("{} -> {1}", i, test.tweet.values[i])
        