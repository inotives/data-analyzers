import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk 

from utils.tools import load_csv_from_data


# Download VADER lexicon if not already installed
# nltk.download('vader_lexicon')

def run_sa():
    df = load_csv_from_data('sample_tweat')

    # Initialize VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis on the 'Tweet Content' column
    df['sentiment'] = df['Tweet Content'].apply(lambda tweet: sia.polarity_scores(tweet)['compound'])

    # Classify sentiment as positive, negative, or neutral
    df['sentiment_label'] = df['sentiment'].apply(lambda score: 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')

    # Display the DataFrame with sentiment analysis
    print(df[['Tweet Content', 'sentiment', 'sentiment_label']])