import sqlite3
import torch
import spacy
import os
import re
import time
from transformers import BertJapaneseTokenizer, BertModel
from tqdm import tqdm

# GPUの利用可能性を確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

wiki_directory = "./wiki"

# GiNZAモデルをロード
nlp = spacy.load("ja_ginza")

def extract_info(text):
    pattern = r'<doc id="(.*?)" url=".*?" title="(.*?)">(.*?)</doc>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def initialize_bert_model():
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    return tokenizer, model

def chunk_text(text, max_bytes=49000):
    bytes_text = text.encode('utf-8')
    start = 0
    chunks = []
    while start < len(bytes_text):
        end = start + max_bytes
        chunk = bytes_text[start:end].decode('utf-8', 'ignore').rsplit('.', 1)[0] + '.'
        chunks.append(chunk)
        start += len(chunk.encode('utf-8'))
    return chunks

def remove_unnecessary_words(text):
    chunks = chunk_text(text)
    filtered_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        unnecessary_pos = ["CONJ", "ADP", "PUNCT"]
        filtered_tokens = [token.text for token in doc if token.pos_ not in unnecessary_pos]
        filtered_chunks.append(' '.join(filtered_tokens))
    return ''.join(filtered_chunks)

def get_word_embedding(tokenizer, model, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
    return embeddings

def initialize_db():
    conn = sqlite3.connect('./wiki_data_analysis/db/words_embeddings_from_mask.db')
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

def is_already_processed(cursor, word):
    cursor.execute("SELECT COUNT(*) FROM word_embeddings WHERE word = ?", (word,))
    return cursor.fetchone()[0] > 0

def save_to_db(cursor, word, wiki_id, content, vector):
    cursor.execute("INSERT OR REPLACE INTO word_embeddings (word, id, content, vector) VALUES (?, ?, ?, ?)", 
                   (word, wiki_id, content, vector.tobytes()))

def main():
    start_time = time.time()

    tokenizer, model = initialize_bert_model()
    conn, cursor = initialize_db()

    total_dirs = sum([len(dirs) for r, dirs, f in os.walk(wiki_directory)])

    with tqdm(total=total_dirs, desc="Overall Progress") as pbar:
        for root, dirs, files in os.walk(wiki_directory):
            for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
                if '.' not in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        extracted_info = extract_info(text)
                        for wiki_id, word, content in extracted_info:
                            if not is_already_processed(cursor, word):
                                wikipedia_text = remove_unnecessary_words(content)
                                vector = get_word_embedding(tokenizer, model, wikipedia_text)
                                save_to_db(cursor, word, wiki_id, content, vector)
                        conn.commit()
                pbar.update(1)

    conn.close()

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
