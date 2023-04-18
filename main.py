from flask import Flask, request
from dotenv import load_dotenv
import tweepy
import os
from flask_cors import CORS
import ml


load_dotenv()

app = Flask(__name__)
CORS(app)


BEARER_TOKEN = os.environ.get('BEARER_TOKEN')
client = tweepy.Client(bearer_token=BEARER_TOKEN)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def get_tweets_by_query(query: str, limit):
    tweets_raw = tweepy.Paginator(client.search_recent_tweets,
                                  query=query,
                                  tweet_fields=['context_annotations', 'created_at'],
                                  max_results=limit
                                  ).flatten(limit=4)

    tweets = []

    for tweet in tweets_raw:

        sentiment = ml.function_1(tweet.text)

        tweets.append({
            "id": tweet.id,
            "sentiment": sentiment.tolist(),
        })

    return tweets


@app.route("/tweets/query", methods=['POST'])
def load_tweet_from_query():
    data = request.json
    _query = data['query']
    limit = data['limit']

    tweets = get_tweets_by_query(_query, limit)

    return tweets


@app.route("/tweets/id", methods=["POST"])
def load_tweet_from_id():
    data = request.json
    tweet_id = data['id']

    tweet = client.get_tweet(tweet_id)

    sentiment = ml.function_1(tweet.data.text)

    return {
        "id": tweet.data.id,
        "sentiment": sentiment.tolist(),
    }


if __name__ == "__main__":
    app.run(debug=True)
