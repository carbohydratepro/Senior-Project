from transformers import AutoTokenizer, AutoModel, BertJapaneseTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# 東北大学が開発した日本語BERTモデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# 東北大学のBERTモデルとトークナイザーのロード
# tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

def compute_similarity(sentence1, sentence2):
    # 文章をトークン化し、モデルに入力できる形式に変換
    inputs1 = tokenizer(sentence1, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
    inputs2 = tokenizer(sentence2, return_tensors='pt', truncation=True, max_length=128, padding='max_length')

    # モデルを使って文章の表現を計算
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # 最初のトークン（[CLS]トークン）の表現を取得
    sentence_embedding1 = outputs1.last_hidden_state[0, 0]
    sentence_embedding2 = outputs2.last_hidden_state[0, 0]

    # コサイン類似度を計算
    similarity = 1 - cosine(sentence_embedding1, sentence_embedding2)

    return similarity

# 2つの文章を定義
sentence1 = "私は犬が好きです。"
sentence2 = "それは機械学習を用いたメガネの学習です。"

# 類似度を計算
similarity = compute_similarity(sentence1, sentence2)
print(f"Similarity: {similarity}")
