import sqlite3
import torch
import requests
import nltk
import time
import spacy
import os
import re
from bs4 import BeautifulSoup
from transformers import BertJapaneseTokenizer, BertModel
from tqdm import tqdm


# ベクトル化の方法を検討

# GPUの利用可能性を確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wiki_directory = "./wiki"

# GiNZAモデルをロード
nlp = spacy.load("ja_ginza")

def extract_info(text):
    # <doc>タグ内のid, title, 本文を抽出する正規表現
    pattern = r'<doc id="(.*?)" url=".*?" title="(.*?)">(.*?)</doc>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

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
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # データをGPUに移動
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()  # データをCPUに移動してからnumpy配列に変換
    return embeddings

def initialize_db():
    conn = sqlite3.connect('./wiki_data_analysis/db/words_embeddings.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS word_embeddings (
        word TEXT PRIMARY KEY,
        id INTEGER,
        content TEXT,
        vector BLOB
    )
    ''')
    return conn, cursor

def save_to_db(cursor, word, wiki_id, content, vector):
    cursor.execute("INSERT OR REPLACE INTO word_embeddings (word, id, content, vector) VALUES (?, ?, ?, ?)", 
                   (word, wiki_id, content, vector.tobytes()))


def main():
    tokenizer, model = initialize_bert_model()
    conn, cursor = initialize_db()

    
    for root, dirs, files in os.walk(wiki_directory):
        for i, file in enumerate(files):
            print(f"processing status : {i+1} / {len(files)}")
            # 拡張子がないファイルを対象とする
            if '.' not in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    extracted_info = extract_info(text)
                    for wiki_id, word, content in tqdm(extracted_info):
                        wikipedia_text = remove_unnecessary_words(content)
                        vector = get_word_embedding(tokenizer, model, wikipedia_text)
                        
                        save_to_db(cursor, word, wiki_id, content, vector)
                    conn.commit()
                        
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
