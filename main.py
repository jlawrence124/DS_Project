from src.processing.preprocess import preprocess_data
from src.processing.theme_analyzer import theme_analyzer_main
from src.visualization.top_words_and_bigrams import plot_data_main

def main():
    """
    Main function.
    """
    company_data_frame_list = preprocess_data()
    for data_frame in company_data_frame_list:
        theme_analyzer_main(data_frame)
    # plot_data_main()

if __name__ == "__main__":
    main()
