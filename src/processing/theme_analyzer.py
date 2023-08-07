from collections import defaultdict
from typing import List

import pandas as pd
import string
import nltk

from src.resources.word_lists import (
    food_related_keywords,
    secondary_yogurt_brand_accounts,
    secondary_yogurt_brands,
    yogurt_brand_accounts,
    yogurt_brand_names,
    yogurt_keywords,
)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("twitter_samples")

from nltk import FreqDist, bigrams
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

custom_stopwords = {
    "rt",
    "via",
    "…",
    "that's",
    "...",
    "’",

}


def tokenize(tweets: List[str]) -> List[List[str]]:
    """
    Tokenize the tweets and remove stop words.
    """
    tokenizer = TweetTokenizer()
    tokenized_tweets = [tokenizer.tokenize(tweet) for tweet in tweets]
    return tokenized_tweets


def filter_out_non_informative_tokens(tweets: List[List[str]]) -> List[List[str]]:
    """
    Filter out non-informative tokens sourced from our word lists resource and stop tokens.
    """
    stop_words = set(stopwords.words("english"))
    stop_words.update(custom_stopwords)

    non_informative_words = set(
        food_related_keywords
        + yogurt_brand_names
        + yogurt_brand_accounts
        + yogurt_keywords
        + secondary_yogurt_brand_accounts
        + secondary_yogurt_brands
    )
    non_informative_words.update(stop_words)
    non_informative_words.update(set(string.punctuation))

    filtered_tweets = [
        [word.lower() for word in tweet if word.lower() not in non_informative_words]
        for tweet in tweets
    ]
    return filtered_tweets


def get_frequency_distribution(filtered_tweets: List[List[str]], company_name: str):
    """
    Getting frequency distribution of individual words and bigrams
    """
    flat_list = [word for sublist in filtered_tweets for word in sublist]

    word_freq = defaultdict(int)
    bigram_freq = defaultdict(int)

    for word in flat_list:
        word_freq[word] += 1
    for bigram in bigrams(flat_list):
        bigram_freq[bigram] += 1

    # Getting top 10 words and bigrams
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    top_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    # Writing to csv
    df_top_words = pd.DataFrame(top_words, columns=['word', 'frequency'])
    df_top_bigrams = pd.DataFrame(top_bigrams, columns=['bigram', 'frequency'])
    df_top_words.to_csv(f'data/processed/companies/{company_name}/top_words.csv', index=False)
    df_top_bigrams.to_csv(f'data/processed/companies/{company_name}/top_bigrams.csv', index=False)


def theme_analyzer_main(tweet_list_df: pd.DataFrame):
    """
    Process list of company data frames and analyze themes.
    """
    filtered_tweets = []

    for tweet in tweet_list_df["text"]:
        tokenized_tweet = tokenize([tweet])
        filtered_tweet = filter_out_non_informative_tokens(tokenized_tweet)
        filtered_tweets.extend(filtered_tweet)

    get_frequency_distribution(filtered_tweets, tweet_list_df["company_name"].iloc[0])
