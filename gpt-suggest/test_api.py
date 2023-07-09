import feedparser
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# arXiv APIを使用してコンピュータ科学分野の最新の論文を取得
url = 'http://export.arxiv.org/rss/cs'
feed = feedparser.parse(url)

# 論文のタイトルを収集
titles = [entry['title'] for entry in feed.entries]

# TF-IDFベクトライザーを作成
vectorizer = TfidfVectorizer(stop_words='english')

# タイトルからTF-IDFベクトルを計算
tfidf = vectorizer.fit_transform(titles)

# 各論文のキーワードを抽出
keywords = []
for vector in tfidf:
    # TF-IDFスコアが最も高い単語を取得
    keyword = vectorizer.get_feature_names()[vector.argmax()]
    keywords.append(keyword)

print(keywords)
