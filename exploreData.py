import re as regularExpressions

import main

def numberOfTwitterUsernameMentions(document):
    twitterUsernameRegex = regularExpressions.compile('\(@?\w{1,15}\)')
    counter = 0
    for word in document.split():
        counter +=len(regularExpressions.findall(twitterUsernameRegex, word))
    return counter

print(main.falseDataNews['text'][0])
