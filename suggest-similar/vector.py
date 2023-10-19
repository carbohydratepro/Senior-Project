# 必要なライブラリのインストール
# !pip install transformers

import torch
from transformers import BertJapaneseTokenizer, BertModel
import numpy as np
import requests
from bs4 import BeautifulSoup
import spacy


# モデルとトークナイザのロード
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def chunk_text(text, max_bytes=49000):
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

def remove_unnecessary_words(text):
    # GiNZAモデルをロード
    nlp = spacy.load("ja_ginza")

    chunks = chunk_text(text)
    filtered_chunks = []
    for chunk in chunks:
        # チャンクごとにトークン化とフィルタリングを行う
        doc = nlp(chunk)
        unnecessary_pos = ["CONJ", "ADP", "PUNCT"]
        filtered_tokens = [token.text for token in doc if token.pos_ not in unnecessary_pos]
        filtered_chunks.append(' '.join(filtered_tokens))

    return ''.join(filtered_chunks)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def similarity_percentage(vec1, vec2):
    sim = cosine_similarity(vec1, vec2)
    # コサイン類似度を0から1の範囲に変換して、0から100の範囲にスケーリング
    return (sim + 1) / 2 * 100



def get_word_embedding(sentence, target_word):
    # トークン化
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # モデルの実行
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 最初のレイヤーの出力を取得
    embeddings = outputs.last_hidden_state[0].numpy()
    
    # target_wordのトークンを取得
    target_tokens = tokenizer.tokenize(target_word)
    if not target_tokens:
        raise ValueError(f"Unable to tokenize word: {target_word}")
    
    # 最初のトークンIDを取得
    token_id = tokenizer.convert_tokens_to_ids(target_tokens[0])
    
    # 最初のトークンの位置を見つける
    idx = (inputs['input_ids'][0] == token_id).nonzero()
    if idx[0].size == 0:
        raise ValueError(f"Token ID for {target_word} ({token_id}) not found in input IDs")
    
    # 対象の単語のベクトルを返す
    return embeddings[idx[0][0]]


def get_wikipedia_paragraphs(word):
    # WikipediaのURLを作成
    url = f"https://ja.wikipedia.org/wiki/{word}"

    # ページの内容を取得
    try:
        response = requests.get(url)
        response.raise_for_status()

        # BeautifulSoupを使用してHTMLを解析
        soup = BeautifulSoup(response.text, 'html.parser')

        # <p>タグの内容を取得
        paragraphs = soup.find_all('p')
        text_content = [p.get_text().strip() for p in paragraphs]

        return '\n\n'.join(text_content)
    except Exception as e:
        print("error")



def main():
    word1 = "帰郷"
    word2 = "動物"
    
    sentence1 = get_wikipedia_paragraphs(word1)
    sentence2 = get_wikipedia_paragraphs(word2)
    
    sentence1 = remove_unnecessary_words(sentence1)
    sentence2 = remove_unnecessary_words(sentence2)
    
    embedding1 = get_word_embedding(sentence1, word1)
    embedding2 = get_word_embedding(sentence2, word2)
    

    sim_percent = similarity_percentage(embedding1, embedding2)
    print(f"Similarity: {sim_percent:.2f}%")


if __name__ == "__main__":
    main()
    
