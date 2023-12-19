from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 日本語テキストのトークン化
def tokenize_japanese(text):
    t = Tokenizer()
    return [token.surface for token in t.tokenize(text)]

# テキストのリスト
texts = ["研究室の論文テキスト1", "提案する論文テキスト2"]  # 実際のテキストに置き換えてください

# TF-IDFベクトル化
vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform(texts)

# コサイン類似度の計算
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print(f"コサイン類似度: {cosine_sim[0][0]}")
