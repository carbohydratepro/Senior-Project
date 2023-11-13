from transformers import BertTokenizer, BertModel
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# BERTの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 単語をトークン化し、BERT入力用のIDに変換
words = ['king', 'man', 'woman', 'queen']
input_ids = tokenizer(words, padding=True, return_tensors='pt')['input_ids']

# モデルに入力して出力を得る
with torch.no_grad():
    outputs = model(input_ids)

# 最終層の隠れ状態を取得
hidden_states = outputs.last_hidden_state

# 単語のベクトルを取得（[CLS]トークンは無視）
word_vectors = hidden_states[:, 1:-1, :]

# ベクトルを単語ごとに平均化（必要であれば）
word_vectors_mean = torch.mean(word_vectors, dim=1)

# t-SNEによる次元削減
tsne = TSNE(n_components=3, random_state=42, perplexity=2)  # 4つのサンプルのためにperplexityを小さく設定
word_vectors_compressed = tsne.fit_transform(word_vectors_mean.detach().numpy())


# 3Dプロットを設定
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 各単語のベクトルをプロット
for word, vec in zip(words, word_vectors_compressed):
    ax.scatter(vec[0], vec[1], vec[2], label=word)

ax.scatter(-68.9, 111.0, 273.0 , label="calc")

# ラベルと凡例を追加
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# プロット表示
plt.show()
