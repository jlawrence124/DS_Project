from pathlib import Path
from typing import List
import pandas as pd
from dotenv import load_dotenv
from sentiment_analysis import analyze

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
    data_frame_list = []
    for csv_file in csv_files:
        data_frame = pd.read_csv(csv_file, encoding="ISO-8859-1")
        data_frame_list.append(data_frame)
    data_frame = pd.concat(data_frame_list, ignore_index=True)
    return data_frame


def write_data_quality_text_file(data_frame: pd.DataFrame):
    """
    Writes a text file with data quality information about the dataset.
    """
    with open(Path("data/processed/data_quality.txt"), "w", encoding="utf-8") as file:
        file.write(f"Column names:\n{data_frame.columns}\n\n")
        file.write(f"Number of rows:\n{data_frame.shape[0]}\n\n")
        file.write(f"Number of null rows:\n{data_frame.isnull().sum()}\n\n")
        file.write(f"Number of duplicate rows:\n{data_frame.duplicated().sum()}\n\n")

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
        file.write(
            f"""highest retweet count entries:\n
            {data_frame.nlargest(5, 'retweet_count')
            .drop(columns=columns_to_remove_from_retweet_data)
            }"""
        )
    file.close()


def write_data_quality_csv_file(data_frame: pd.DataFrame):
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
                data_frame.shape[0],
                data_frame.isnull().sum().sum(),
                data_frame.duplicated().sum(),
                round(
                    (
                        data_frame[data_frame["retweeted"] == "TRUE"].shape[0]
                        / data_frame.shape[0]
                    )
                    * 100,
                    2,
                ),
            ],
        }
    )
    csv_path = Path("data/processed/statistics.csv")
    statistics_df.to_csv(csv_path, index=False)


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
    analyze(combined_data_frame)


if __name__ == "__main__":
    preprocess_data()
