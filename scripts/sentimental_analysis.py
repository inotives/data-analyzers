import pandas as pd 
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import plotly.express as px

pd.set_option('max_colwidth', None) # show full width of showing cols

# Download lexicon if not already installed
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---- Sentimental Analysis Main Code ------------------------------------------------------------------

def preprocess_text(text):
    """Preprocessing text """
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [word.lower() for word in tokens]
    
    # Remove stop words and non-alphabetic characters
    tokens = [word for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def apply_sentiment_analysis(data, content_col):
    """Apply VADER sentiment analysis and return the updated DataFrame."""
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis on the 'content' column and store all sentiment scores
    data['sentiment_score'] = data[content_col].apply(lambda content: sia.polarity_scores(content)['compound'])
    data['positive_score'] = data[content_col].apply(lambda content: sia.polarity_scores(content)['pos'])
    data['negative_score'] = data[content_col].apply(lambda content: sia.polarity_scores(content)['neg'])
    data['neutral_score'] = data[content_col].apply(lambda content: sia.polarity_scores(content)['neu'])

    # Classify sentiment as positive, negative, or neutral based on the compound score
    data['sentiment_label'] = data['sentiment_score'].apply(lambda score: 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')

    return data

def generate_ngrams(data, content_col, ngram_range=(2, 2)):
    """Generate N-Grams from the content column."""
    ngram_vectorizer = CountVectorizer(ngram_range=ngram_range)
    ngrams = ngram_vectorizer.fit_transform(data[content_col])
    
    # Convert N-Grams to DataFrame
    ngram_df = pd.DataFrame(ngrams.toarray(), columns=ngram_vectorizer.get_feature_names_out(), index=data.index)
    
    return ngram_df


def apply_tfidf(data, content_col):
    """Apply TF-IDF on the content column."""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[content_col])
    
    # Convert TF-IDF to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=data.index)
    
    return tfidf_df


def train_sentiment_model(features, labels):
    """Train a sentiment classifier."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Using Logistic Regression for demonstration
    model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    return model


def perform_sa(data, content_col):
    """Main function to perform sentiment analysis, generate N-Grams, apply TF-IDF, and predict sentiment."""
    # Handle missing data
    data = data.dropna(subset=[content_col])

    # Preprocess text
    data['content'] = data[content_col].apply(preprocess_text)

    # Apply VADER sentiment analysis
    data = apply_sentiment_analysis(data, 'content')

    # Generate N-Grams
    ngram_df = generate_ngrams(data, 'content', ngram_range=(2, 2))

    # Apply TF-IDF
    tfidf_df = apply_tfidf(data, 'content')

    # Combine VADER scores, N-Gram, and TF-IDF features
    combined_features = pd.concat([ngram_df, tfidf_df, data[['positive_score', 'negative_score', 'neutral_score']]], axis=1)
    
    # Train a model to predict sentiment labels
    model = train_sentiment_model(combined_features, data['sentiment_label'])
    
    # Use the trained model to predict sentiment labels
    data['predicted_sentiment'] = model.predict(combined_features)
    
    return data
