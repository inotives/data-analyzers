NEWS_ARTICLES_CNT = """
SELECT 
    na.source,
    count(*) as cnt
FROM news_articles na 
GROUP BY 1
ORDER by 1 DESC
;
"""

NEWS_ARTICLES_LAST_7_DAYS = """
SELECT 
    na.uniq_key,
    na.article_date,
    CONCAT(na.title, '. ', na.content) as content_text
FROM news_articles na
WHERE na.article_date >= NOW() - INTERVAL '7 days'
ORDER BY 2 ASC
"""

CREATE_NEWS_SENTIMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS news_sentiments (
    id SERIAL PRIMARY KEY,
    news_id VARCHAR(255) NOT NULL,
    content TEXT,
    sentiment_score NUMERIC,
    updated_at TIMESTAMP
);
"""