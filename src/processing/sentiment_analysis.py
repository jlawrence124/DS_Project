import logging
import os
from dataclasses import dataclass
from typing import List

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
    tweet_text: str = ""
    score: float = 0.0
    label: str = ""
    brand_risk: bool = False


def analyze(data_frame: pd.DataFrame, batch_size: int=50, company_name: str="") -> pd.DataFrame:
    """
    Analyzes the sentiment of a given text by a set batch size.
    """
    # Reset the index of the data frame
    copied_data_frame = data_frame.copy().reset_index(drop=True)
    raw_tweets_text = get_raw_tweet_text_data(data_frame)
    sentiments = []
    for i in tqdm(
        range(0, len(raw_tweets_text), batch_size), desc="Analyzing sentiments"
    ):
        batch = raw_tweets_text[i : i + batch_size]
        result = sentiment_analyzer(batch)

        sentiment_raw = []
        sentiment_scores = []
        sentiment_labels = []
        # pair up the batch with the result
        for tweet, res in zip(batch, result):
            sentiment_raw.append(Sentiment(tweet, float(res["score"]), res["label"]))
            sentiment_scores.append(res["score"])
            sentiment_labels.append(res["label"])
        end_index = i + len(batch) - 1

        copied_data_frame.loc[i : end_index, "sentiment_score"] = sentiment_scores
        copied_data_frame.loc[i : end_index, "sentiment"] = sentiment_labels
        sentiments.extend(sentiment_raw)

    write_company_data_frame_to_csv(copied_data_frame, company_name)
    return copied_data_frame

def get_raw_tweet_text_data(data_frame: pd.DataFrame) -> List[str]:
    """
    Get all the text data for all tweets.
    """
    return data_frame["text"].tolist()

def write_company_data_frame_to_csv(
    filtered_data_frame: pd.DataFrame,
    company_name_snake_case: str,
):
    """
    Writes a csv file with sentiment analysis of the dataset.
    """
    directory_path = f"data/processed/companies/{company_name_snake_case}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    filtered_data_frame_path = (
        f"{directory_path}/{company_name_snake_case}_relevant_tweets.csv"
    )
    filtered_data_frame.to_csv(filtered_data_frame_path, index=False, encoding="utf-8")
