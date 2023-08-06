from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import pandas as pd
from sentiment_analysis import analyze


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

    file_name_key_word = brand_name_lower.replace(" ", "_")
    company_keywords = company_twitter_handles + [brand_name_lower] + alternate_names

    combined_keywords = []
    yogurt_keywords = [
        "yogurt",
        "yoghurt",
        "yoghourt",
        "pro-biotic",
        "probiotic",
        brand_name,
    ]
    food_related_keywords = [
        "delicious",
        "tasty",
        "healthy",
        "breakfast",
        "dairy",
        "flavor",
        "creamy",
        "creamery",
        "protein",
        "spoonful",
        "snack",
        "dessert",
        "nutritious",
    ]
    yogurt_brand_names = [
        "activia",
        "chobani",
        "dannon",
        "fage",
        "greek gods",
        "liberte",
        "maple hill",
        "noosa",
        "organic valley",
        "siggi",
        "smari",
        "stonyfield",
        # "vanilla bean",
        "wallaby",
        "yoplait",
    ]
    yogurt_brand_accounts = [
        "@activia",
        "@activiauk",
        "@chobani",
        "@chobani_uk",
        "@dannon",
        "@fageusa",
        "@fageuk",
        "@fage_fr",
        "@thegreekgods",
        "@greekgodsuk",
        "@liberteusa",
        "@libertecanada",
        "@maplehillcream",
        "@noosayoghurt",
        "@organicvalley",
        "@siggisdairy",
        "@smariyogurt",
        "@smariorganics",
        "@stonyfield",
        "@wallabyyogurt",
        "@yoplait",
    ]
    other_yogurt_brands = [
        "brown cow",
        "cabot",
        "lactalis",
        "oikos",
        "powerful yogurt",
        "yocrunch",
    ]
    other_yogurt_brand_accounts = [
        "@browncowyogurt",
        "@cabotcheese",
        "@cabotcreamery",
        "@groupe_lactalis",
        "@oikos",
        "@lovemysilk",
        "@powerfulyogurt",
        "@uk_lactalis",
        "@yocrunch",
    ]

    combined_keywords = (
        yogurt_keywords
        + company_keywords
        + yogurt_brand_names
        + yogurt_brand_accounts
        + other_yogurt_brands
        + other_yogurt_brand_accounts
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
                    or any(word in x.lower() for word in other_yogurt_brands)
                    or any(word in x.lower() for word in other_yogurt_brand_accounts)
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

    relevant_tweets_file_path = (
        f"data/processed/{file_name_key_word}_relevant_tweets.csv"
    )
    relevant_tweets.to_csv(relevant_tweets_file_path, index=False, encoding="utf-8")

    # if the filtered data_frame is over the specified threshold
    if len(relevant_tweets) >= relevancy_threshold:
        return data_frame

    print(f"BELOW RELEVANCY THRESHOLD for {brand_name} tweets. Filtering out...\n\n")

    return data_frame[~data_frame["file"].str.contains(file_name_key_word)]


def prepare_data_for_filtering(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares brand dict to filter out of dataset if they do not have relevant yogurt tweets.
    """
    # brand dict with twitter handles, company name, and alternate names
    brands: Dict[str, Brand] = {
        "Activia": Brand(
            twitter_handles=["@activia", "@activiauk"],
            brand_name="Activia",
            negative_keywords=["activia benz", "mens-rights-activia"],
        ),
        "Chobani": Brand(
            twitter_handles=["@chobani", "@chobani_uk"],
            brand_name="Chobani",
        ),
        "Dannon": Brand(
            twitter_handles=["@dannon"],
            brand_name="Dannon",
            alternate_names=[
                "danone",
            ],
            is_nonspecific_name=True,
        ),
        "Fage": Brand(
            twitter_handles=["@fageusa", "@fageuk"],
            brand_name="Fage",
        ),
        "Greek Gods": Brand(
            twitter_handles=["@thegreekgods", "@greekgodsuk"],
            brand_name="Greek Gods",
            is_nonspecific_name=True,
        ),
        "Liberte": Brand(
            twitter_handles=["@liberteusa", "@libertecanada"],
            brand_name="Liberte",
            alternate_names=[
                "liberté",
            ],
            is_nonspecific_name=True,
        ),
        "Maple Hill": Brand(
            twitter_handles=["@maplehillcream"],
            brand_name="Maple Hill",
            is_nonspecific_name=True,
        ),
        "Noosa": Brand(
            twitter_handles=["@noosayoghurt"],
            brand_name="Noosa",
        ),
        "Organic Valley": Brand(
            twitter_handles=["@OrganicValley"],
            brand_name="Organic Valley",
        ),
        "Siggi": Brand(
            twitter_handles=["@siggisdairy"],
            brand_name="Siggi",
        ),
        "Smari": Brand(
            twitter_handles=["@smariyogurt", "@smariorganics"],
            brand_name="Smari",
            alternate_names=[
                "smári",
            ],
        ),
        "Stonyfield": Brand(
            twitter_handles=["@stonyfield"],
            brand_name="Stonyfield",
        ),
        "Wallaby": Brand(
            twitter_handles=["@wallabyyogurt"],
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
            twitter_handles=["@yoplait"],
            brand_name="Yoplait",
        ),
    }

    print(f"\n\nBefore filtering ::: {len(data_frame)}")

    # this will filter out both greek_gods json file entries
    for values in brands.values():
        data_frame = remove_tweets_with_negative_keywords(data_frame, values)
        data_frame = filter_irrelevant_data(
            data_frame,
            brand=values,
            relevancy_threshold=5,
        )
    print(f"After filtering ::: {len(data_frame)}\n\n")

    return data_frame


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
    write_data_quality_text_file(combined_data_frame)
    write_data_quality_csv_file(combined_data_frame)

    # remove twitter links - regex test here https://regex101.com/r/wZ0dAP/1
    combined_data_frame["text"] = combined_data_frame["text"].str.replace(
        r"http[s]?://t\.[^\s]*|[^[$]]", "", regex=True
    )

    # drop duplicate tweets
    # deduped_data_frame = combined_data_frame.drop_duplicates(subset=["text"])

    # filter out irrelevant data
    filtered_data_frame = prepare_data_for_filtering(combined_data_frame)

    print(len(filtered_data_frame))

    # send reduced size data_frame to sentiment analysis
    # analyze(filtered_and_deduped_data_frame.head(1000), 20)
    # analyze(filtered_and_deduped_data_frame)


if __name__ == "__main__":
    preprocess_data()
