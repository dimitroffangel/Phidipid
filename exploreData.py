import matplotlib.pyplot
import pandas
import re as regularExpressions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

import main
import newsDataHelper
import visualizerHelper

def numberOfTwitterUsernameMentions(document):
    twitterUsernameRegex = regularExpressions.compile(r'@([A-Za-z0-9_]+)')
    return len(regularExpressions.findall(twitterUsernameRegex, document))

print(main.falseDataNews['text'][0])
listOfNumberOfTweets = pandas.DataFrame(columns=['class', 'numberOfTweets'])
for _, row in main.allNewsData.iterrows():
    listOfNumberOfTweets=pandas.concat([listOfNumberOfTweets, pandas.DataFrame({"class":[row['class']], 'numberOfTweets':[numberOfTwitterUsernameMentions(row['text'])]})], ignore_index=True)

sumTweetsInFalseNews = 0
for _, row in main.falseDataNews.iterrows():
    sumTweetsInFalseNews += numberOfTwitterUsernameMentions(row['text'])


sumTweetsInTrueNews = 0
for _, row in main.trueDataNews.iterrows():
    sumTweetsInTrueNews += numberOfTwitterUsernameMentions(row['text'])

print("number of Tweets in False news", sumTweetsInFalseNews)
print("number of Tweets in True news", sumTweetsInTrueNews)
# visualizerHelper.plotData(listOfNumberOfTweets)

numberOfUniqueWordsInFalseNews = newsDataHelper.countUniqueWords(main.falseDataNews['text'].values)
numberOfUniqueWordsInTrueNews = newsDataHelper.countUniqueWords(main.trueDataNews['text'].values)

falseNewsTokenization = newsDataHelper.removeNoisyData(main.falseDataNews['text'].values, len(main.falseDataNews['text'].values))
numberOfUniqueWordsInFalseNewsPostTokenization = newsDataHelper.countUniqueWords(falseNewsTokenization)

trueNewsTokenization = newsDataHelper.removeNoisyData(main.trueDataNews['text'].values, len(main.trueDataNews['text'].values))
numberOfUniqueWordsInTrueNewsPostTokenization = newsDataHelper.countUniqueWords(trueNewsTokenization)

uniqueWordsFalseDataNews = newsDataHelper.wordsFrequency(falseNewsTokenization)[:100]
uniqueWordsTrueDataNews = newsDataHelper.wordsFrequency(trueNewsTokenization)[:100]

visualizerHelper.plotListOfTuples(uniqueWordsFalseDataNews[:21])
visualizerHelper.plotListOfTuples(uniqueWordsTrueDataNews[:21])

numberOfWords = len(main.allNewsData['text'].values)
corpus = newsDataHelper.removeNoisyData(main.allNewsData['text'].values, numberOfWords)
countVectorizer = CountVectorizer(max_features=500)
X = countVectorizer.fit_transform(corpus).toarray()
chi2Score = chi2(X, main.allNewsData['text'])
chi2PFeaturesStatistics = chi2Score[0]
chi2PValues = chi2Score[1]
dependentChiFeatures = list(zip(countVectorizer.get_feature_names(), chi2PFeaturesStatistics))
visualizerHelper.plotListOfTuples(uniqueWordsFalseDataNews)