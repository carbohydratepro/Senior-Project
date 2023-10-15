import requests
from bs4 import BeautifulSoup

def get_wikipedia_paragraphs(word):
    # WikipediaのURLを作成
    url = f"https://ja.wikipedia.org/wiki/{word}"

    # ページの内容を取得
    response = requests.get(url)
    response.raise_for_status()

    # BeautifulSoupを使用してHTMLを解析
    soup = BeautifulSoup(response.text, 'html.parser')

    # <p>タグの内容を取得
    paragraphs = soup.find_all('p')
    text_content = [p.get_text().strip() for p in paragraphs]

    return '\n\n'.join(text_content)

# 例の実行
word = "猫"
paragraphs_content = get_wikipedia_paragraphs(word)
print(paragraphs_content[:1000])  # 最初の1000文字だけ表示
