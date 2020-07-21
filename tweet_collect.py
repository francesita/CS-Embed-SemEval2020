import tweepy
import pickle
import sys
import json


#dic = {}

consumer_token = open( "private/.consumer_token" ).read().strip()
consumer_secret = open("private/.consumer_secret").read().strip()

access_token = open("private/.access_token").read().strip()
access_token_secret = open("private/.access_secret").read().strip()

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# this constructs API's instance                                                                                                                                                                                   
api = tweepy.API(auth, wait_on_rate_limit = True)

#collected tweets are saved in a list of tweet objects
tweetObjects=[]
#json file where tweets will be saved to
updated_dic_file = open("cs_tweets.json", "w", encoding='utf-8')
#Keywords used as query to extract tweets
kw_file = open("cs_kw.txt", "r")


i = 0
def find_tweet(kw_file):
    '''
    Function takes kw_file as parameter. For each word in the file, the extract_tweet function is called to extract all tweets with the associated keyword. Then all tweets are saved onto the json file to     have all data saved incase of any issues. 
    '''
    for line in kw_file:
        q = line
        extract_tweet(q)
        tweets = remove_duplicates(tweetObjects)
        json.dump(tweets, updated_dic_file, indent=2)

def remove_duplicates(tweetObjects):
    '''
    Function checks for duplicates in tweetObjects 
    '''
    tweets = []
    tweetIds = []
    for tweet in tweetObjects:
        if tweet['tweet_id'] not in tweetIds:
            tweetIds.append(tweet['tweet_id']) 
            tweets.append(tweet)
        else:
            continue
    return tweets

def extract_tweet(query):
    '''
    Function takes a word as parameter, to use as query and extract tweets related to query
    '''
	print(query)
	try:
		for tweet in tweepy.Cursor(api.search, q=query, lang="en",tweet_mode = "extended", status="2020-06-25").items(5000):			
			if hasattr(tweet, 'retweeted_status'):
				tweet_id = tweet.retweeted_status.id
				tweet_date = tweet.retweeted_status.created_at
				tweet_author = tweet.retweeted_status.author._json['screen_name']
				tweet_text = tweet.retweeted_status.full_text
				tweet_place = tweet.retweeted_status.author._json['location']
			else:
				tweet_id = tweet.id
				tweet_date = tweet.created_at
				tweet_author = tweet.author._json['screen_name']
				tweet_text = tweet.full_text
				tweet_place = tweet.author._json['location']


			#puttinh all info in a dic and appending to list tweetObject        		
			tweetObjects.append({
				'tweet_id': tweet_id,
				'author': tweet_author,
				'tweet': tweet_text,
				'location': tweet_place
			})		

	except tweepy.TweepError as e:
		print(e.reason)
		print(query)

find_tweet(kw_file)








