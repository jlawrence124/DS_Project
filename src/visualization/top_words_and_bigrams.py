from typing import List, Tuple
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

def read_bigram_csv_data_from_file(company_name: str) -> pd.DataFrame:
    """
    Reads the bigram csv data from file.
    """
    return pd.read_csv(f'data/processed/companies/{company_name}/top_bigrams.csv')

def read_top_word_csv_data_from_file(company_name: str) -> pd.DataFrame:
    """
    Reads the word csv data from file.
    """
    return pd.read_csv(f'data/processed/companies/{company_name}/top_words.csv')


def plot_data(top_words: List[Tuple[str, int]], top_bigrams: List[Tuple[Tuple[str, str], int]], company_name: str):
    words, word_freq = zip(*top_words)
    plt.bar(words, word_freq)
    plt.title(f'Top Words for {company_name}')
    plt.savefig(f'data/processed/companies/{company_name}/top_words.png', bbox_inches='tight')

    bigrams, bigram_freq = zip(*top_bigrams)
    plt.bar(bigrams, bigram_freq)
    plt.title(f'Top Bigrams for {company_name}')
    plt.savefig(f'data/processed/companies/{company_name}/top_bigrams.png', bbox_inches='tight')


def plot_data_main():
    """
    Main function.
    """
    for company_name in Path('data/processed/companies').glob('*'):
        company_name = str(company_name).split('/')[-1]
        df1 = read_top_word_csv_data_from_file(company_name)
        df2 = read_bigram_csv_data_from_file(company_name)
        top_words = zip(df1['word'], df1['frequency'])
        top_bigrams = zip(df2['bigram'], df2['frequency'])
        plot_data(top_words, top_bigrams, company_name)