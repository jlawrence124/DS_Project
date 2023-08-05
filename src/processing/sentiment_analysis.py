import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal GPU) device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

pipe = pipeline("text-classification", model=MODEL_NAME)

sentiment_analyzer = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=device,
    max_length=512,
    truncation=True,
)

@dataclass
class Sentiment:
    """
    Sentiment object
    """
    tweet_text=""
    score=0.0
    label=""
    brand_risk=False


def analyze(data_frame: pd.DataFrame, batch_size=50) -> List[Sentiment]:
    """
    Analyzes the sentiment of a given text by a set batch size.
    """
    copied_data_frame = data_frame.copy()
    raw_tweets_text = get_raw_tweet_text_data(data_frame)
    sentiments = []
    for i in tqdm(
        range(0, len(raw_tweets_text), batch_size), desc="Analyzing sentiments"
    ):
        batch = raw_tweets_text[i : i + batch_size]
        result = sentiment_analyzer(batch)
        print(result)

        sentiment_raw = []
        sentiment_scores = []
        sentiment_labels = []
        # pair up the batch with the result
        for tweet, res in zip(batch, result):
            sentiment_raw.append(Sentiment(tweet, float(res["score"]), res["label"]))
            sentiment_scores.append(res["score"])
            sentiment_labels.append(res["label"])
        copied_data_frame.loc[i : i + len(batch) - 1, "sentiment_score"] = sentiment_scores
        copied_data_frame.loc[i : i + len(batch) - 1, "sentiment"] = sentiment_labels
        sentiments.extend(sentiment_raw)

    print(sentiments[0:10] + ["..."] + sentiments[-10:])
    # copied_data_frame["sentiment"] = sentiments
    write_sentiment_to_csv(copied_data_frame)
    return sentiments


def get_raw_tweet_text_data(data_frame: pd.DataFrame) -> List[str]:
    """
    Get all the text data for all tweets.
    """
    return data_frame["text"].tolist()


def write_sentiment_to_csv(data_frame: pd.DataFrame):
    """
    Writes a csv file with sentiment analysis of the dataset.
    """
    data_frame.to_csv(Path("data/processed/sentiment_small.csv"), index=False, encoding="utf-8")
