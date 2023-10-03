from read_csv import read_csv_files, read_csv_info, data_output
import logging
import pandas as pd
import spacy
from spacy import displacy
from janome.tokenizer import Tokenizer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ログの設定
logging.basicConfig(level=logging.INFO)

# GiNZAモデルをロード
nlp = spacy.load("ja_ginza")

# BERTモデルとトークナイザーをロード
bert_tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

def extract_pos(text, target_pos):
    doc = nlp(text)
    extracted_words = [token.text for token in doc if token.pos_ == target_pos]
    return extracted_words

def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return list(set(nouns))

def get_bert_embedding(word):
    inputs = bert_tokenizer(word, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # 平均プーリング

def find_most_similar(word, text):
    word_embedding = get_bert_embedding(word)
    # 特定の単語を除外
    exclude_words = {"よう", "こと"}
    nouns = [noun for noun in extract_nouns(text) if noun not in exclude_words]
    print(nouns)
    
    similarity_scores = []
    
    for noun in nouns:
        noun_embedding = get_bert_embedding(noun)
        sim = cosine_similarity(word_embedding.detach().numpy(), noun_embedding.detach().numpy())[0][0]
        similarity_scores.append((noun, sim))
    
    # 類似度スコアでソートし、上位5件を取得
    top_5_words = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]
    
    return top_5_words


def main():
    logging.info("start program...")
    contents = pd.read_csv("./theme-decision-support/data/contents.csv").iloc[:,0].tolist()
    
    text = contents[198]
    word = "検証"
    
    print(word)
    # target_pos = "NOUN"  # 名詞を抽出する場合
    # result = extract_pos(text, target_pos)
    

    # 最も類似した単語を見つける
    top_5_words = find_most_similar(word, text)
    for idx, (word, score) in enumerate(top_5_words, start=1):
        print(f"{idx}. Word: {word}, Similarity: {score:.4f}")
        
    for idx, (word, score) in enumerate(top_5_words, start=1):
        print(f"　& {word} & {score:.4f} \\\\")
        
        
if __name__ == "__main__":
    main()



