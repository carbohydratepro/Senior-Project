from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from janome.tokenizer import Tokenizer

# Janomeを使用して日本語をトークナイズ
tokenizer = Tokenizer()

# ダミーデータ
sentences_ja = [
    ["猫", "動物"],
    ["犬", "動物"],
    ["魚", "動物"],
    ["車", "乗り物"],
    ["バス", "乗り物"],
    ["ラップトップ", "機械"],
]

# トークナイズ
tokenized_sentences = [list(tokenizer.tokenize(' '.join(sent), wakati=True)) for sent in sentences_ja]
 
# Word2Vecモデルの学習
model_ja = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
model_ja.train(tokenized_sentences, total_examples=model_ja.corpus_count, epochs=10)

# 単語のベクトルを取得
words_ja = list(model_ja.wv.index_to_key)
vectors_ja = [model_ja.wv[word] for word in words_ja]

# K-meansクラスタリング
kmeans_ja = KMeans(n_clusters=3, random_state=0).fit(vectors_ja)
labels_ja = kmeans_ja.labels_

# クラスタごとに単語をグループ化
clusters_ja = {}
for word, label in zip(words_ja, labels_ja):
    if label not in clusters_ja:
        clusters_ja[label] = []
    clusters_ja[label].append(word)

print(clusters_ja)
