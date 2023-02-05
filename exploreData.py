import pandas
import re as regularExpressions

import main
import visualizerHelper

def numberOfTwitterUsernameMentions(document):
    twitterUsernameRegex = regularExpressions.compile(r'@([A-Za-z0-9_]+)')
    counter = 0
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