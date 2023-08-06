from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, bigrams
from collections import defaultdict

# Sample data provided
sample_data = [
    "RT @barntiques859: Maple Hill Dairy Milk Bottle Round Red Pyro",
    "RT @MapleHillCream: Maple Hill's average milking herd is only 55 cows, leading to more active pasture management. #BetterforCows #BetterFor",
    "@patrickhinds @MapleHillCream @Chobani @Stonyfield I await @Chobani 'a follow back post generous offer (whole milk plain please)",
    "I wanna use twitter to get free greek yogurt. Who's it gonna be? @MapleHillCream ? @Chobani ? @Stonyfield ?",
]

# Tokenizing the tweets
tokenized_tweets = [word_tokenize(tweet.lower()) for tweet in sample_data]

# Removing stop words
stop_words = set(stopwords.words('english'))
filtered_tweets = [[word for word in tweet if word not in stop_words] for tweet in tokenized_tweets]

# Removing non-informative words
non_informative_words = {'yogurt', 'dairy', 'rt'}
filtered_tweets = [[word for word in tweet if word not in non_informative_words] for tweet in filtered_tweets]

# Getting frequency distribution of individual words and bigrams
word_freq = defaultdict(int)
bigram_freq = defaultdict(int)

for tweet in filtered_tweets:
    for word in tweet:
        word_freq[word] += 1
    for bigram in bigrams(tweet):
        bigram_freq[bigram] += 1

# Getting top 10 words and bigrams
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
top_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:10]

top_words, top_bigrams
