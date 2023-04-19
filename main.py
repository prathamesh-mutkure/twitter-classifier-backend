from flask import Flask, request
from dotenv import load_dotenv
import tweepy
import os
from flask_cors import CORS
import ml
from flask_sqlalchemy import SQLAlchemy
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secret'

db = SQLAlchemy(app)

BEARER_TOKEN = os.environ.get('BEARER_TOKEN')
client = tweepy.Client(bearer_token=BEARER_TOKEN)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))


class Tweet(db.Model):
    tweet_id = db.Column(db.String, primary_key=True)
    sentiment = db.Column(db.JSON)


with app.app_context():
    db.create_all()


def store_tweet_if_not_exists(tweet_id, tweet_sentiment):
    existing_tweet = Tweet.query.filter_by(tweet_id=tweet_id).first()

    if not existing_tweet:

        sentiment_json = json.dumps(tweet_sentiment)

        tweet = Tweet(tweet_id=tweet_id, sentiment=sentiment_json)

        db.session.add(tweet)
        db.session.commit()

        return True
    
    return False


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/signup", methods=["POST"])
def signup():
    username = request.form['username']
    password = request.form['password']

    if not username or not password:
        return {
            "error": "All Fields are compulsoty"
        }, 400

    user = User.query.filter_by(username=username).first()

    if user:
        return {
            "error": "Username is already taken"
        }, 400

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    return {
        "id": new_user.id,
        "username": new_user.username,
    }

@app.route("/signin", methods=['POST'])
def login():

    print(request.form);

    username = request.form['username']
    password = request.form['password']
        
    user = User.query.filter_by(username=username).first()
        
    if user and (user.password == password):
        return {
            "id": user.id,
            "username": user.username,
        }

    else:
        return {
            "error": "Invalid Credentials"
        }


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
            "id": str(tweet.id),
            "sentiment": sentiment.tolist()[0],
        })

    return tweets


@app.route("/tweets/query", methods=['POST'])
def load_tweet_from_query():
    data = request.json
    _query = data['query']
    limit = data['limit']

    tweets = get_tweets_by_query(_query, limit)

    for tweet in tweets:
        store_tweet_if_not_exists(tweet_id=tweet["id"],tweet_sentiment=tweet["sentiment"])

    return tweets


@app.route("/tweets/id", methods=["POST"])
def load_tweet_from_id():
    data = request.json
    tweet_id = data['id']

    tweet = client.get_tweet(tweet_id)

    sentiment = ml.function_1(tweet.data.text)

    tweet_new = {
        "id": str(tweet.data.id),
        "sentiment": sentiment.tolist()[0],
    }

    store_tweet_if_not_exists(tweet_id=tweet_new["id"],tweet_sentiment=tweet_new["sentiment"])

    return tweet_new


if __name__ == "__main__":
    app.run(debug=True)
