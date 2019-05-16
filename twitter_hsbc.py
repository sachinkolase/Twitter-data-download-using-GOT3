
apt-get install python3-urllib3 

try:
    import urllib.request as urllib2
except ImportError:
    import urllib2




#import urllib2
from bs4 import BeautifulSoup


url = "https://twitter.com/search?q=akparti%20since%3A2015-05-01%20until%3A2015-06-05&amp;amp;amp;amp;amp;amp;lang=en"
response = urllib2.urlopen(url)
html = response.read()
soup = BeautifulSoup(html)


tweets = soup.find_all('li','js-stream-item')
for tweet in tweets:
    if tweet.find('p','tweet-text'):
        tweet_user = tweet.find('span','username').text
        tweet_text = tweet.find('p','tweet-text').text.encode('utf8')
        tweet_id = tweet['data-item-id']
        timestamp = tweet.find('a','tweet-timestamp')['title']
        tweet_timestamp = dt.datetime.strptime(timestamp, '%H:%M - %d %b %Y')
    else:
        continue
        
    

tweets


import GetOldTweets3

max_tweets = 150

import csv
csvFile = open('result.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

tweetCriteria = GetOldTweets3.manager.TweetCriteria().setSince("2019-01-01").setUntil("2019-04-30").setQuerySearch("hsbc uk together we thrive").setMaxTweets(max_tweets).setNear("London").setWithin("500mi")

tweetCriteria1 = GetOldTweets3.manager.TweetCriteria().setSince("2019-01-01").setUntil("2019-04-30").setQuerySearch("hsbc uk together we thrive").setMaxTweets(max_tweets)

tweetCriteria2 = GetOldTweets3.manager.TweetCriteria().setQuerySearch("hsbc uk together we thrive").setMaxTweets(max_tweets)


for i in range(max_tweets):
    tweet = GetOldTweets3.manager.TweetManager.getTweets(tweetCriteria2)[i]
    print(tweet.id)
    print(tweet.username)
    print(tweet.text)
    print(tweet.date)
    print(tweet.geo)
    csvWriter.writerow([tweet.date, tweet.text.encode('utf-8')])
    print (tweet.date, tweet.text)
csvFile.close()
  
import pandas
result2 = pandas.read_csv('result.csv')
result2 = result2.iloc[:-6]
print(result)  

result2.columns = ['Date', 'Text']




# Performing Sentiment Analysis
import textblob
from textblob import TextBlob 
textblob.download_corpora

import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline


#removing hashtags
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    

result2['tweet'] = np.vectorize(remove_pattern)(result2['Text'], "@[\w]*")


#Removing Punctuations, Numbers, and Special Characters

result2['tweet'] = result2['tweet'].str.replace("[^a-zA-Z#]", " ")

       
#removing short words

result2['tweet'] = result2['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

#tockenization
tokenized_tweet = result2['tweet'].apply(lambda x: x.split())
tokenized_tweet.head()


#stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
tokenized_tweet.head()


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

result2['tweet'] = tokenized_tweet


all_words = ' '.join([text for text in result2['tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
