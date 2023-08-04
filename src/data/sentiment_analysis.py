from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
MODEL_NAME = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
)

sentiment_analyzer = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=device
)


@dataclass
class Sentiment:
    """
    Sentiment object
    """

    score: float
    label: str
    brand_risk: bool = False


def analyze(data_frame: pd.DataFrame, batch_size=50) -> List[Sentiment]:
    """
    Analyzes the sentiment of a given text by a set batch size.
    """
    results = []
    raw_tweets_text = get_raw_tweet_text_data(data_frame)
    for i in tqdm(
        range(0, len(raw_tweets_text), batch_size), desc="Analyzing sentiments"
    ):
        batch = raw_tweets_text[i : i + batch_size]
        result = sentiment_analyzer(batch)
        results.extend(result)
    return results


def get_raw_tweet_text_data(data_frame: pd.DataFrame) -> List[str]:
    """
    Get all the text data for all tweets.
    """
    return data_frame["text"].tolist()


def write_sentiment_to_csv(data_frame: pd.DataFrame):
    """
    Writes a csv file with sentiment analysis of the dataset.
    """

    data_frame["sentiment"] = data_frame["text"].apply(analyze)
    data_frame.to_csv(Path("data/processed/sentiment.csv"), index=False)
