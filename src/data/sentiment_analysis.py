from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")


class Sentiment:
    def __init__(self, score: float, label: str, brand_risk: bool = False):
        self.score = score
        self.label = label
        self.brand_risk = brand_risk


def analyze(text: str):
    result = sentiment_analyzer(text)
    for item in result:
        sentiment = Sentiment(score=item["score"], label=item["label"])

    return sentiment
