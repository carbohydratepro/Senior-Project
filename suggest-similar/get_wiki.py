import requests
from bs4 import BeautifulSoup
from googletrans import Translator

def get_wikipedia_paragraphs(word):
    # 英語WikipediaのURLを作成
    en_url = f"https://en.wikipedia.org/wiki/{word}"
    
    # 日本語WikipediaのURLを作成
    ja_url = f"https://ja.wikipedia.org/wiki/{word}"
    
    def fetch_content(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text_content = [p.get_text().strip() for p in paragraphs]
            return '\n\n'.join(text_content), None
        except Exception as e:
            return None, str(e)
    
    # 英語のWikipediaでページの内容を取得
    content, error = fetch_content(en_url)
    if content:
        # 翻訳オブジェクトを作成
        translator = Translator()
        
        # 英文を日本語に翻訳
        translated_text = translator.translate(content, src='en', dest='ja').text
        return translated_text, None
    
    # 日本語のWikipediaでページの内容を取得
    content, error = fetch_content(ja_url)
    if content:
        return content, None
    
    return None, "Failed to fetch content from both English and Japanese Wikipedia."

# サンプルの単語でテスト
word = "Python"
content, error = get_wikipedia_paragraphs(word)
print(content if content else error)
