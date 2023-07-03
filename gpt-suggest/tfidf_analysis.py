from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer

# 日本語のテキスト文書 (例として2文)
documents = [
    "私の名前はジョン・ドウです。私はプログラミングが好きです。",
    "彼女の名前はジェーン・ドウです。彼女は数学が好きです。"
]

# Janomeのトークナイザーをインスタンス化
t = Tokenizer(wakati=True)

# TF-IDFベクトルライザーのインスタンス化
vectorizer = TfidfVectorizer(tokenizer=t.tokenize, use_idf=True, smooth_idf=True)

# 文書のベクトル化
X = vectorizer.fit_transform(documents)

# 文書ごとに単語とそのTF-IDFスコアを表示
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    # 単語とそのTF-IDFスコアを格納する辞書を作成
    word2tfidf = {word: tfidf for word, tfidf in zip(vectorizer.get_feature_names_out(), X[i].toarray()[0])}
    # スコアで降順にソート
    sorted_word2tfidf = sorted(word2tfidf.items(), key=lambda x: x[1], reverse=True)
    # 上位10単語とそのスコアを表示
    for word, score in sorted_word2tfidf[:10]:
        print(f"    {word}: {score}")
    print("\n")
