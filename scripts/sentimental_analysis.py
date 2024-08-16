import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

from utils.tools import load_csv_from_data, export_data_to_csv


# Download VADER lexicon if not already installed
# nltk.download('vader_lexicon')

def run_sa():
    data = load_csv_from_data('_OUTPUT_cointelegraph_news')

    df = perform_sa(data, 'Title')

    visualize_sentiment(df)

    export_data_to_csv(df, 'SA_cointelegraph_news')


def perform_sa(data, content_col):   

    # Handle missing data
    data = data.dropna(subset=[content_col])

    # generalize the content col to content 
    data = data.rename(columns={content_col:'content'})

    # Initialize VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis on the 'content' column
    data['sentiment_scores'] = data['content'].apply(lambda content: sia.polarity_scores(content))
    data['sentiment_score'] = data['sentiment_scores'].apply(lambda x: x['compound'])
    data['positive_score'] = data['sentiment_scores'].apply(lambda x: x['pos'])
    data['negative_score'] = data['sentiment_scores'].apply(lambda x: x['neg'])
    data['neutral_score'] = data['sentiment_scores'].apply(lambda x: x['neu'])

    # Classify sentiment as positive, negative, or neutral
    data['sentiment_label'] = data['sentiment_score'].apply(lambda score: 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')

    print(data[['content', 'sentiment_score', 'positive_score', 'negative_score', 'neutral_score', 'sentiment_label']])

    return data

def visualize_sentiment(data):
    # Plot sentiment distribution
    fig = px.bar(
        data_frame=data,
        x='sentiment_label',
        title="Sentiment Analysis Results",
        labels={'sentiment_label': 'Sentiment', 'count': 'Count'},
        color='sentiment_label'
    )
    fig.show()

    # Print data to review
    print(data[['content', 'sentiment_score', 'sentiment_label']])