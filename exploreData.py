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
visualizerHelper.plotData(listOfNumberOfTweets)

numberOfUniqueWordsInFalseNews = newsDataHelper.countUniqueWords(main.falseDataNews['text'].values)
numberOfUniqueWordsInTrueNews = newsDataHelper.countUniqueWords(main.trueDataNews['text'].values)

numberOfUniqueWordsInFalseNewsPostTokenization = newsDataHelper.countUniqueWords(
    newsDataHelper.removeNoisyData(
        main.falseDataNews['text'].values, 
        len(main.falseDataNews['text'].values)
    )
)
numberOfUniqueWordsInTrueNewsPostTokenization = newsDataHelper.countUniqueWords(
    newsDataHelper.removeNoisyData(
        main.trueDataNews['text'].values, 
        len(main.trueDataNews['text'].values)
    )
)

uniqueWordsFalseDataNews = newsDataHelper.wordsFrequency(main.falseDataNews['text'].values)[:100]
uniqueWordsTrueDataNews = newsDataHelper.wordsFrequency(main.trueDataNews['text'].values)[:100]