import sqlite3
import torch
import requests
import nltk
import time
import spacy
import os
import numpy as np
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
from transformers import BertJapaneseTokenizer, BertModel
from tqdm import tqdm
from googletrans import Translator
from mask import predict_masked_token


# GPUの利用可能性を確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_bert_model():
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)  # モデルをGPUに移動
    return tokenizer, model


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


def get_word_embedding(tokenizer, model, sentence):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # データをGPUに移動
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()  # データをCPUに移動してからnumpy配列に変換
        return embeddings, None
    except Exception as e:
        return None, str(e)


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


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def similarity_percentage(vec1, vec2):
    sim = cosine_similarity(vec1, vec2)
    # コサイン類似度を0から1の範囲に変換して、0から100の範囲にスケーリング
    return (sim + 1) / 2 * 100


    
def find_sim(target_vector, rows):
    word_sim = []
    for i, row in enumerate(tqdm(rows)):
        word, vector_blob = row
        vector = np.frombuffer(vector_blob, dtype=np.float32)
        sim_percent = similarity_percentage(target_vector, vector)
        word_sim.append([word, sim_percent])

    sorted_data = sorted(word_sim, key=lambda x: x[1], reverse=True)
    top_10 = sorted_data[:10]
    return top_10

def return_vector(target_word, tokenizer, model):
    wikipedia_text, wiki_error = get_wikipedia_paragraphs(target_word)
    if wikipedia_text:
        wikipedia_text = remove_unnecessary_words(wikipedia_text)
        target_vector, bert_error = get_word_embedding(tokenizer, model, wikipedia_text)
        return target_vector
    else:
        target_vector, bert_error = None, None
        return None
    

def create_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS difference_vectors (
            id INTEGER PRIMARY KEY,
            word1 TEXT NOT NULL,
            word2 TEXT NOT NULL,
            vector BLOB NOT NULL
        )
    ''')

def store_difference_vector(cursor, word1, word2, diff_vector):
    vector_blob = diff_vector.tobytes()
    cursor.execute("INSERT INTO difference_vectors (word1, word2, vector) VALUES (?, ?, ?)",
                   (word1, word2, vector_blob))
    
    
# ベクトル演算を行う関数
def vector_arithmetic(vector1, vector2, eq):
    if eq == 'plus':
        vector = vector1 + vector2
    elif eq == 'minas':
        vector = vector1 - vector2
    else:
        print('calcerror')
        exit()
    return vector
    
def main():
    def calc_diff_pairs(cursor, rows):
        for i in tqdm(range(len(rows))):
            for j in range(len(rows)):
                vec1 = np.frombuffer(rows[i][1], dtype=np.float32)
                vec2 = np.frombuffer(rows[j][1], dtype=np.float32)
                diff_vector = vec1 - vec2
                
                # 差分ベクトルをデータベースに保存
                store_difference_vector(cursor, rows[i][0], rows[j][0], diff_vector)

            # 変更をコミット
            conn.commit()


    tokenizer, model = initialize_bert_model()
    
    # Connect to the database
    conn = sqlite3.connect('./suggest-similar/db/words_embeddings.db')
    cursor = conn.cursor()

    # Fetch all the records where vector is not NULL
    cursor.execute("SELECT word1, word2, vector FROM difference_vectors")
    rows = cursor.fetchall()
    
    # データベースを閉じる
    conn.close()
    
    
    word1, word2 = "機械学習", "計算"
    
    vector1, vector2 = return_vector(word1, tokenizer, model), return_vector(word2, tokenizer, model)
    
    target_vector = vector_arithmetic(vector1, vector2, 'minas')

    top_10 = find_sim(target_vector, rows)
    for result in top_10:
        word, per = result
        print(f"{word}:{per}")

    target_vector = vector_arithmetic(vector2, vector1, 'minas')

    top_10 = find_sim(target_vector, rows)
    for result in top_10:
        word, per = result
        print(f"{word}:{per}")
        
if __name__ == "__main__":
    # 差分の計算
    main()


# メディアパイプ　オープンポーズ
# RNN LSTM