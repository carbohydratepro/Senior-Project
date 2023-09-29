from janome.tokenizer import Tokenizer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Janomeトークナイザーを初期化
janome_tokenizer = Tokenizer()

# BERTモデルとトークナイザーをロード
bert_tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

def get_bert_embedding(word):
    inputs = bert_tokenizer(word, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # 平均プーリング

def find_most_similar(word, text):
    word_embedding = get_bert_embedding(word)
    janome_tokens = janome_tokenizer.tokenize(text, wakati=True)  # wakatiモードでトークン化
    
    similarity_scores = []
    
    for token in janome_tokens:
        token_embedding = get_bert_embedding(token)
        sim = cosine_similarity(word_embedding.detach().numpy(), token_embedding.detach().numpy())[0][0]
        similarity_scores.append((token, sim))
    
    # 類似度スコアでソートし、上位5件を取得
    top_5_words = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]
    
    return top_5_words

# 入力単語と文
input_word = "edit"
input_text = "一連の手話動作画像からOpenPoseを用いて関節座標を取得する"

# 最も類似した単語を見つける
top_5_words = find_most_similar(input_word, input_text)
for idx, (word, score) in enumerate(top_5_words, start=1):
    print(f"{idx}. Word: {word}, Similarity: {score:.4f}")
