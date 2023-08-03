import glob
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List

# Load the environment variables from the .env file
load_dotenv()

def get_csv_files() -> List[str]:
    """
    Returns a list of all csv files in the data/raw directory.
    """
    csv_files = glob.glob("data\\raw\\*.csv")
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
    with open("data\\processed\\data_quality.txt", "w") as f:
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
    statistics_df.to_csv("data\\processed\\statistics.csv", index=False)


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

if __name__ == "__main__":
    preprocess_data()
