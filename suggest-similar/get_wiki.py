import requests
from bs4 import BeautifulSoup
from googletrans import Translator

def chunk_text(text, max_bytes=5000):
    """テキストをバイトサイズに基づいてチャンクに分割する"""
    bytes_text = text.encode('utf-8')
    start = 0
    chunks = []
    while start < len(bytes_text):
        end = start + max_bytes
        # バイト列を文字列にデコードして、最後の完全な文を見つける
        chunk = bytes_text[start:end].decode('utf-8', 'ignore').rsplit('.', 1)[0] + '.'
        chunks.append(chunk)
        start += len(chunk.encode('utf-8'))
    return chunks


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
        
        chunks = chunk_text(content)
        filtered_chunks = []
        for chunk in chunks:
            translated_text = translator.translate(chunk, src='en', dest='ja').text
            filtered_chunks.append(''.join(translated_text))

        return ''.join(filtered_chunks), None
    
    # 日本語のWikipediaでページの内容を取得
    content, error = fetch_content(ja_url)
    if content:
        return content, None
    
    return None, "Failed to fetch content from both English and Japanese Wikipedia."

# サンプルの単語でテスト
word = "日本語"
content, error = get_wikipedia_paragraphs(word)
print(content if content else error)
