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
    capitalized_brand_name="",
    company_keywords=[],
    relevancy_threshold=5,
) -> pd.DataFrame:
    """
    Returns a filtered DataFrame if the input DataFrame is full of overwhelmingly irrelevant tweets.
    """
    file_name_key_word = capitalized_brand_name.lower().replace(" ", "_")

    combined_keywords = []
    yogurt_keywords = [
        "yogurt",
        "yoghurt",
        "yoghourt",
        capitalized_brand_name,
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
    # keep these lowercase
    yogurt_brand_names = [
        "activia",
        "chobani",
        "dannon",
        "fage",
        "greek gods",
        "liberté",
        "maple hill",
        "noosa",
        "organic valley",
        "siggi",
        "smári",
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
        "@yocrunch",
    ]

    combined_keywords.extend(yogurt_keywords)

    # add the other yogurt brands to the list of keywords
    combined_keywords.extend(company_keywords)
    combined_keywords.extend(yogurt_brand_names)
    combined_keywords.extend(yogurt_brand_accounts)
    combined_keywords.extend(other_yogurt_brands)
    combined_keywords.extend(other_yogurt_brand_accounts)

    # Because "Vanilla Bean" is food related, we can not rely on food adjective keywords
    if capitalized_brand_name not in ["Vanilla Bean"]:
        combined_keywords.extend(food_related_keywords)

    # initialize relevant_tweets as an empty DataFrame
    relevant_tweets = pd.DataFrame()

    # If the brand name is a common word, unaffiliated name or brand,
    # we need to at least have another brand mention or yogurt keyword
    if capitalized_brand_name in ["Greek Gods", "Siggi", "Liberté",  "Wallaby", "Vanilla Bean",]:
        # remove the current brand name from the list of yogurt brands
        yogurt_brand_names.remove(capitalized_brand_name.lower())
        relevant_tweets = data_frame[
            data_frame["text"].apply(
                lambda x: capitalized_brand_name.lower() in x.lower() and (
                    # if the tweet mentions any yogurt keywords
                    any(word in x.lower() for word in yogurt_keywords)
                    # or if the tweet mentions any food related keywords
                    or any(word in x.lower() for word in food_related_keywords)
                    # or if the tweet mentions another yogurt brand
                    or any(word in x.lower() for word in yogurt_brand_names)
                    or any(word in x.lower() for word in yogurt_brand_accounts)
                    or any(word in x.lower() for word in other_yogurt_brands)
                    or any(word in x.lower() for word in other_yogurt_brand_accounts)
                )
            )
        ]
    else:
        relevant_tweets = data_frame[
            (data_frame["text"].str.contains(capitalized_brand_name)
                | data_frame["text"].apply(
                    lambda x: any(word in x.lower() for word in company_keywords)
                )
            )
            & data_frame["text"].apply(
                lambda x: any(word in x.lower() for word in combined_keywords)
            )
        ]
    print(
        f"Number of relevant {capitalized_brand_name} tweets found: {len(relevant_tweets)}"
    )

    relevant_tweets_file_path = (
        f"data/processed/{file_name_key_word}_relevant_tweets.csv"
    )
    relevant_tweets.to_csv(relevant_tweets_file_path, index=False, encoding="utf-8")

    # if the filtered data_frame is over the specified threshold
    if len(relevant_tweets) >= relevancy_threshold:
        return data_frame

    print(
        f"BELOW RELEVANCY THRESHOLD for {capitalized_brand_name} tweets. Filtering out...\n\n"
    )

    return data_frame[~data_frame["file"].str.contains(file_name_key_word)]


def prepare_data_for_filtering(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares list of brands to filter out of dataset if they do not have relevant yogurt tweets.
    """
    brand_keywords = {
        "Activia": ["activia", "@activia", "@activiauk",],
        "Chobani": ["chobani", "@chobani", "@chobani_uk",],
        "Dannon": ["dannon", "@dannon", "danone",],
        "Fage": ["fage", "@fageusa", "@fageuk",],
        "Greek Gods": ["greek_gods", "@thegreekgods", "@greekgodsuk",],
        "Liberté": ["liberte", "liberté", "@liberteusa", "@libertecanada",],
        "Maple Hill": ["maple_hill", "@maplehillcream",],
        "Noosa": ["noosa", "@noosayoghurt",],
        "Organic Valley": ["organic_valley", "@OrganicValley",],
        "Siggi": ["siggi", "@siggisdairy",],
        "Smari": ["smari", "@smariyogurt", "@smariorganics", "smári",],
        "Stonyfield": ["stonyfield", "@stonyfield",],
        # leaving out vanilla bean for the time being as it muddies the data
        # "Vanilla Bean": ["vanilla_bean",],
        "Wallaby": ["wallaby", "@wallabyyogurt",],
        "Yoplait": ["yoplait", "@yoplait",],
    }

    print(f"\n\nBefore filtering ::: {len(data_frame)}")
    # this will filter out both greek_gods json file entries
    for key, values in brand_keywords.items():
        data_frame = filter_irrelevant_data(data_frame, key, values, 5)
    print(f"After filtering ::: {len(data_frame)}\n\n")

    return data_frame


def preprocess_data():
    """
    Calls other functions to preprocess the data.
    """
    csv_list = get_csv_files()

    combined_data_frame = combine_csv_data(csv_list)
    write_data_quality_text_file(combined_data_frame)
    write_data_quality_csv_file(combined_data_frame)

    # remove twitter links - regex test here https://regex101.com/r/wZ0dAP/1
    combined_data_frame["text"] = combined_data_frame["text"].str.replace(r"http[s]?://t\.[^\s]*|[^[$]]", "", regex=True)

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
