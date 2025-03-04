import os
from dotenv import load_dotenv
import tweepy
import requests

load_dotenv()

# twitter_client = tweepy.Client(
...
# )


def scrape_user_tweets(username, num_tweets=5, mock: bool = False):
    """
    Scrape a user's tweets from Twitter
    """
    
    tweet_list = []
    
    if mock:
        EDEN_TWITTER_GIST = "https://gist.githubusercontent.com/emarco177/9d4fdd52dc432c72937c6e383dd1c7cc/raw/1675c4b1595ec0ddd8208544a4f915769465ed6a/eden-marco-tweets.json"
        tweets = requests.get(EDEN_TWITTER_GIST, timeout=10).json()
        
        for tweet in tweets:
            tweet_dict = {}
            tweet_dict["text"] = tweet["text"]
            tweet_dict["url"] = f"https://x.com/{username}/status/{tweet['id']}"
            tweet_list.append(tweet_dict)
            
        return tweet_list
    
    else:
        raise NotImplementedError("Scraping tweets from Twitter is not implemented yet")
        
    
if __name__ == "__main__":
    tweets = scrape_user_tweets(username="EdenEmarco177", mock=True)
    print(tweets)
