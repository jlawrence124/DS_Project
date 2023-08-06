from typing import List    

def get_yogurt_keywords(brand_name: str) -> List[str]:
    """
    Returns a list of yogurt keywords with an appended brand name.
    """
    return [
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
secondary_yogurt_brands = [
    "brown cow",
    "cabot",
    "lactalis",
    "oikos",
    "powerful yogurt",
    "yocrunch",
]
secondary_yogurt_brand_accounts = [
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