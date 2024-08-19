import scripts.sentimental_analysis as sa 
import utils.sql_list as sql
from utils.db import DBConnection
from utils.settings import POSTGRES_DB_URL




def perform_sentiment_on_news_articles():
    conn = DBConnection(POSTGRES_DB_URL)
    conn.create_engine()

    news_df = conn.pull_data_to_dataframe(sql.NEWS_ARTICLES_LAST_7_DAYS)

    news_sentiment = sa.perform_sa(news_df,'content_text')

    print(news_sentiment.columns)
    
    conn.close()