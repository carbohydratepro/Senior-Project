import urllib.request
import feedparser

# arXiv APIのエンドポイントと検索クエリ
base_url = 'http://export.arxiv.org/api/query?'
search_query = 'cat:cs.AI'  # AIカテゴリーの論文を検索
start = 0                    # 取得開始位置
max_results = 10             # 取得する結果の数

query = f'{base_url}search_query={search_query}&start={start}&max_results={max_results}'
response = urllib.request.urlopen(query).read()

# レスポンスをパース
feed = feedparser.parse(response)

# 結果を表示
for entry in feed.entries:
    print('Title: ', entry.title)
    print('Link: ', entry.link)
    print('Published: ', entry.published)
    print('Summary: ', entry.summary)
    print('Authors: ', ', '.join(author.name for author in entry.authors))
    print('\n')
