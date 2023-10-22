import sqlite3
import torch
import requests
import nltk
import time
import spacy
import os
import numpy as np
import csv
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
from transformers import BertJapaneseTokenizer, BertModel
from tqdm import tqdm


# GPUの利用可能性を確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_bert_model():
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)  # モデルをGPUに移動
    return tokenizer, model



def main():
    # Connect to the database
    conn = sqlite3.connect('./suggest-similar/db/words_embeddings.db')
    cursor = conn.cursor()

    # Fetch all the records where vector is not NULL
    cursor.execute("SELECT word, vector FROM word_embeddings WHERE vector IS NOT NULL")
    rows = cursor.fetchall()
    
    word_vector = []
    for i, row in enumerate(tqdm(rows)):
        word, vector_blob = row
        vector = np.frombuffer(vector_blob, dtype=np.float32)
        word_vector.append([word, vector])

    # Save the results to a CSV file
    csv_file = "./suggest-similar/word_vectors.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "vector"])  # Writing headers
        writer.writerows(word_vector)
        
if __name__ == "__main__":
    main()



