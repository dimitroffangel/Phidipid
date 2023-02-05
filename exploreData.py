import matplotlib.pyplot
import pandas
import re as regularExpressions

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