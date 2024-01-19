# 必要なライブラリをインポート
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sentence-BERTモデルをロード（ここでは日本語版モデルを指定）
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens')

# ベクトル化したい文書（例）
documents = ["これは最初の文書です。", "これは二番目の文書です。"]

# 文書をベクトル化
document_embeddings = model.encode(documents)

# コサイン類似度を計算
cosine_sim = cosine_similarity(document_embeddings)

# 結果を出力
print("コサイン類似度:")
print(cosine_sim)

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(cos_sim(document_embeddings[0], document_embeddings[1]))