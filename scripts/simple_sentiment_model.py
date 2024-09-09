import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def data_preprocessing(text):
    ''' preprocess the data '''

    processed_text = text.lower()

    return processed_text

def perform_sa(data, content_col):
    '''Main code that perform the Sentiment Analysis'''
    data = data.dropna(subset=[content_col])

    data['content'] = data[content_col].apply(data_preprocessing)

    content_text = data[['content']]

    vect = TfidfVectorizer(max_features=1000, ngram_range=(1,3)).fit(content_text.content)
    X = vect.transform(content_text.content)

    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())

    print(X_df)



