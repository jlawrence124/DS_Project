import glob
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List
from pathlib import Path

# Load the environment variables from the .env file
load_dotenv()


def get_csv_files() -> List[str]:
    """
    Returns a list of all csv files in the data/raw directory.
    """
    csv_files = [str(file) for file in Path("data/raw").glob("*.csv")]
    print(csv_files)
    return csv_files


def combine_csv_data(csv_files: List[str]) -> pd.DataFrame:
    """
    Combines all csv files into a single pandas dataframe.
    """
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="ISO-8859-1")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def write_data_quality_text_file(df: pd.DataFrame):
    """
    Writes a text file with data quality information about the dataset.
    """
    with open(Path("data/processed/data_quality.txt"), "w") as f:
        f.write(f"Column names:\n{df.columns}\n\n")
        f.write(f"Number of rows:\n{df.shape[0]}\n\n")
        f.write(f"Number of null rows:\n{df.isnull().sum()}\n\n")
        f.write(f"Number of duplicate rows:\n{df.duplicated().sum()}\n\n")

        columns_to_remove_from_retweet_data = [
            "lang",
            "created_at",
            "created_day",
            "timeonly",
            "created_dateonly",
            "datetime",
            "coordinates",
            "geo",
            "place",
            "truncated",
            "user_favourites_count",
            "user_following",
            "user_friends_count",
            "user_geo_enabled",
            "user_listed_count",
            "user_location",
            "user_statuses_count",
            "user_time_zone",
            "file",
        ]
        f.write(
            f"highest retweet count entries:\n{df.nlargest(5, 'retweet_count').drop(columns=columns_to_remove_from_retweet_data)}"
        )
    f.close()


def write_data_quality_csv_file(df: pd.DataFrame):
    """
    Writes a csv file with data quality information about the dataset.
    """
    statistics_df = pd.DataFrame(
        {
            "metric": [
                "total number of rows",
                "number of null rows",
                "number of duplicate rows",
                "percentage that are retweets",
            ],
            "count": [
                df.shape[0],
                df.isnull().sum().sum(),
                df.duplicated().sum(),
                round((df[df["retweeted"] == "TRUE"].shape[0] / df.shape[0]) * 100, 2),
            ],
        }
    )
    csv_path = Path("data/processed/statistics.csv")
    statistics_df.to_csv(csv_path, index=False)

def get_raw_tweet_text_data(df: pd.DataFrame) -> List[str]:
    return df['text'].tolist()

def preprocess_data():
    """
    Calls other functions to preprocess the data.
    """
    csv_list = get_csv_files()
    
    combined_data_frame = combine_csv_data(csv_list)
    write_data_quality_text_file(combined_data_frame)
    write_data_quality_csv_file(combined_data_frame)

    print(combined_data_frame.head(5))
    print(combined_data_frame.shape)
    raw_tweet_list = get_raw_tweet_text_data(combined_data_frame)

    # Write the raw tweet data to a text file or create it if it doesn't exist
    with open(Path("data/processed/tweets.txt"), "w") as f:
        for tweet in raw_tweet_list:
            f.write(f"{tweet}\n\n")
    f.close()


if __name__ == "__main__":
    preprocess_data()
