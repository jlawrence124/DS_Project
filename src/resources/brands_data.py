from dataclasses import dataclass, field
from typing import Dict, List


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