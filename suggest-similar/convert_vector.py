import sqlite3
import torch
import requests
import nltk
import time
import spacy
import os
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
from transformers import BertJapaneseTokenizer, BertModel
from tqdm import tqdm


# ベクトル化の方法を検討

# GPUの利用可能性を確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_progress(index):
    with open("./suggest-similar/progress.txt", "w") as f:
        f.write(str(index))

def load_progress():
    if os.path.exists("./suggest-similar/progress.txt"):
        with open("./suggest-similar/progress.txt", "r") as f:
            return int(f.read().strip())
    return 0

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

def initialize_db():
    conn = sqlite3.connect('./suggest-similar/db/words_embeddings.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS word_embeddings (
        word TEXT PRIMARY KEY,
        vector BLOB,
        error TEXT
    )
    ''')
    return conn, cursor

def save_to_db(cursor, word, vector, error):
    cursor.execute("INSERT OR REPLACE INTO word_embeddings (word, vector, error) VALUES (?, ?, ?)", 
                   (word, vector.tobytes() if vector is not None else None, error))

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

        return '\n\n'.join(text_content), None
    except Exception as e:
        return None, str(e)


def main():
    tokenizer, model = initialize_bert_model()
    conn, cursor = initialize_db()
    
    done_num = int(load_progress())
    
    # WordNetからすべてのlemma（単語）を取得
    all_lemmas = list(wn.all_lemma_names(lang="jpn"))[done_num:]

    BATCH_SIZE = 100  # 100単語を処理するごとにコミット
    for i, lemma in enumerate(tqdm(all_lemmas)):
        wikipedia_text, wiki_error = get_wikipedia_paragraphs(lemma)
        if wikipedia_text:
            wikipedia_text = remove_unnecessary_words(wikipedia_text)
            vector, bert_error = get_word_embedding(tokenizer, model, wikipedia_text)
        else:
            vector, bert_error = None, None
        combined_error = "; ".join(filter(None, [wiki_error, bert_error]))
        save_to_db(cursor, lemma, vector, combined_error)
        
        # 一定数の単語を処理したらコミットと進捗の表示
        if i % BATCH_SIZE == 0:
            conn.commit()
            save_progress(done_num+i)
            
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
