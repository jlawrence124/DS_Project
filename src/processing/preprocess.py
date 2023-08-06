from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import pandas as pd
from sentiment_analysis import analyze
from resources.word_lists import (
    get_yogurt_keywords,
    food_related_keywords,
    yogurt_brand_names,
    yogurt_brand_accounts,
    secondary_yogurt_brands,
    secondary_yogurt_brand_accounts,
)


@dataclass
class Brand:
    """
    Brand object
    """

    twitter_handles: List[str]
    brand_name: str
    alternate_names: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    is_nonspecific_name: bool = False
    has_food_related_name: bool = False


def get_csv_files() -> List[str]:
    """
    Returns a list of all csv files in the data/raw directory.
    """
    csv_files = [str(file) for file in Path("data/raw").glob("*.csv")]
    return csv_files


def combine_csv_data(csv_files: List[str]) -> pd.DataFrame:
    """
    Combines all csv files into a single pandas dataframe.
    """
    data_frames = []
    for file in csv_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as file:
            data_frames.append(pd.read_csv(file))
    data_frame = pd.concat(data_frames, ignore_index=True)
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

        data_frame.describe().to_csv("data/processed/data_describe.csv")
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


def filter_irrelevant_data(
    data_frame: pd.DataFrame,
    brand: Brand,
    relevancy_threshold=5,
) -> pd.DataFrame:
    """
    Returns a filtered DataFrame if the input DataFrame is full of overwhelmingly irrelevant tweets.
    """
    # initialize relevant_tweets as an empty DataFrame
    relevant_tweets = pd.DataFrame()
    company_twitter_handles = brand.twitter_handles
    brand_name = brand.brand_name
    alternate_names = brand.alternate_names
    brand_name_lower = brand_name.lower()

    company_name_snake_case = brand_name_lower.replace(" ", "_")
    company_keywords = company_twitter_handles + [brand_name_lower] + alternate_names

    combined_keywords = []
    yogurt_keywords = get_yogurt_keywords(brand_name_lower)

    combined_keywords = (
        yogurt_keywords
        + company_keywords
        + yogurt_brand_names
        + yogurt_brand_accounts
        + secondary_yogurt_brands
        + secondary_yogurt_brand_accounts
        + (food_related_keywords if brand.has_food_related_name else [])
    )

    if not brand.has_food_related_name:
        combined_keywords.extend(food_related_keywords)

    # If the brand name is a common word, unaffiliated name or brand,
    # we need to at least have another brand mention or yogurt keyword
    if brand.is_nonspecific_name:
        # remove the current brand name from the list of yogurt brands
        yogurt_brand_names.remove(brand_name_lower)
        relevant_tweets = data_frame[
            data_frame["text"].apply(
                lambda x: (
                    brand_name_lower in x.lower()
                    or any(word in x.lower() for word in company_keywords)
                )
                and (
                    # if the tweet mentions any yogurt keywords
                    (
                        any(word in x.lower() for word in yogurt_keywords)
                        # or if the tweet mentions any food related keywords
                        or any(word in x.lower() for word in food_related_keywords)
                    )
                    # if the tweet mentions another yogurt brand
                    or any(word in x.lower() for word in yogurt_brand_names)
                    or any(word in x.lower() for word in yogurt_brand_accounts)
                    or any(word in x.lower() for word in secondary_yogurt_brands)
                    or any(word in x.lower() for word in secondary_yogurt_brand_accounts)
                )
            )
        ]
    else:
        relevant_tweets = data_frame[
            (
                data_frame["text"].str.contains(brand_name)
                | data_frame["text"].apply(
                    lambda x: any(word in x.lower() for word in company_keywords)
                )
            )
            & data_frame["text"].apply(
                lambda x: any(word in x.lower() for word in combined_keywords)
            )
        ]
    print(f"Number of relevant {brand_name} tweets found: {len(relevant_tweets)}")

    # if the filtered data_frame is over the specified threshold
    if len(relevant_tweets) >= relevancy_threshold:
        return relevant_tweets

    print(f"BELOW RELEVANCY THRESHOLD for {brand_name} tweets. Filtering out...\n\n")

    return relevant_tweets[
        ~relevant_tweets["file"].str.contains(company_name_snake_case)
    ]


def prepare_data_for_filtering(data_frame: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Prepares brand dict to filter out of dataset if they do not have relevant yogurt tweets.
    """
    company_data_frame_list = []

    # brand dict with twitter handles, company name, and alternate names
    brands: Dict[str, Brand] = {
        "Activia": Brand(
            twitter_handles=[
                "@activia",
                "@activiauk",
            ],
            brand_name="Activia",
            negative_keywords=[
                "activia benz",
                "mens-rights-activia",
            ],
        ),
        "Chobani": Brand(
            twitter_handles=[
                "@chobani",
                "@chobani_uk",
            ],
            brand_name="Chobani",
        ),
        "Dannon": Brand(
            twitter_handles=[
                "@dannon",
            ],
            brand_name="Dannon",
            alternate_names=[
                "danone",
            ],
            is_nonspecific_name=True,
        ),
        "Fage": Brand(
            twitter_handles=[
                "@fageusa",
                "@fageuk",
            ],
            brand_name="Fage",
        ),
        "Greek Gods": Brand(
            twitter_handles=[
                "@thegreekgods",
                "@greekgodsuk",
            ],
            brand_name="Greek Gods",
            is_nonspecific_name=True,
        ),
        "Liberte": Brand(
            twitter_handles=[
                "@liberteusa",
                "@libertecanada",
            ],
            brand_name="Liberte",
            alternate_names=[
                "liberté",
            ],
            is_nonspecific_name=True,
        ),
        "Maple Hill": Brand(
            twitter_handles=[
                "@maplehillcream",
            ],
            brand_name="Maple Hill",
            is_nonspecific_name=True,
        ),
        "Noosa": Brand(
            twitter_handles=[
                "@noosayoghurt",
            ],
            brand_name="Noosa",
        ),
        "Organic Valley": Brand(
            twitter_handles=[
                "@OrganicValley",
            ],
            brand_name="Organic Valley",
        ),
        "Siggi": Brand(
            twitter_handles=[
                "@siggisdairy",
            ],
            brand_name="Siggi",
        ),
        "Smari": Brand(
            twitter_handles=[
                "@smariyogurt",
                "@smariorganics",
            ],
            brand_name="Smari",
            alternate_names=[
                "smári",
                "#SMARI",
            ],
            is_nonspecific_name=True,
        ),
        "Stonyfield": Brand(
            twitter_handles=[
                "@stonyfield",
            ],
            brand_name="Stonyfield",
        ),
        "Wallaby": Brand(
            twitter_handles=[
                "@wallabyyogurt",
            ],
            brand_name="Wallaby",
            is_nonspecific_name=True,
        ),
        # Vanilla Bean is muddying the data and is likely not even a brand
        # "Vanilla Bean": Brand(
        #     twitter_handles=[],
        #     brand_name="Vanilla Bean",
        #     alternate_names=[],
        #     is_nonspecific_name=True,
        #     has_food_related_name=True,
        # ),
        "Yoplait": Brand(
            twitter_handles=[
                "@yoplait",
            ],
            brand_name="Yoplait",
        ),
    }

    print(f"\n\nBefore filtering ::: {len(data_frame)}")

    for values in brands.values():
        data_frame = remove_tweets_with_negative_keywords(data_frame, values)
        filtered_data_frame = filter_irrelevant_data(
            data_frame,
            brand=values,
            relevancy_threshold=0,
        )
        company_data_frame_list.append(filtered_data_frame)
        # send filtered data_frame to sentiment analysis
        analyze(
            data_frame=filtered_data_frame,
            company_name=values.brand_name.lower().replace(" ", "_"),
        )
    print(f"After filtering ::: {len(company_data_frame_list)}\n\n")

    combined_filtered_data_frame = pd.concat(
        company_data_frame_list, ignore_index=True
    ).drop_duplicates()
    combined_filtered_data_frame.to_csv(
        "data/processed/combined_filtered_data_frame.csv", index=False
    )
    write_data_quality_text_file(combined_filtered_data_frame)
    write_data_quality_csv_file(combined_filtered_data_frame)

    return company_data_frame_list


def remove_tweets_with_negative_keywords(
    data_frame: pd.DataFrame, brand: Brand
) -> pd.DataFrame:
    """
    Removes tweets with negative keywords from the dataset.
    """
    negative_keywords = brand.negative_keywords
    if len(negative_keywords) == 0:
        return data_frame
    negative_keywords_regex = "|".join(negative_keywords).lower()
    return data_frame[
        ~data_frame["text"]
        .str.lower()
        .str.contains(negative_keywords_regex, case=False)
    ]


def preprocess_data():
    """
    Calls other functions to preprocess the data.
    """
    csv_list = get_csv_files()

    combined_data_frame = combine_csv_data(csv_list)

    # remove twitter links - regex test here https://regex101.com/r/wZ0dAP/1
    combined_data_frame["text"] = combined_data_frame["text"].str.replace(
        r"http[s]?://t\.[^\s]*|[^[$]]", "", regex=True
    )

    # filter out irrelevant data
    filtered_data_frame_list = prepare_data_for_filtering(combined_data_frame)

    # for data_frame in filtered_data_frame_list:
    #     # print(data_frame.head())
    #     analyze(data_frame)


if __name__ == "__main__":
    preprocess_data()
