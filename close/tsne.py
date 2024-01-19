import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib
from transformers import BertJapaneseTokenizer, BertModel
import torch
import numpy as np

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)

# CSVファイルのパス
path_titles = './theme-decision-support/data/titles.csv'
path_contents = './theme-decision-support/data/overview.csv'

# Pandasを使用してCSVファイルからデータを読み込む
titles = pd.read_csv(path_titles, header=None, skiprows=1).iloc[:, 0]
contents = pd.read_csv(path_contents, header=None, skiprows=1).iloc[:, 0]

# ランダムに30個の要素を選択
indices = random.sample(range(len(titles)), 20)
selected_titles = titles.iloc[indices]
selected_contents = contents.iloc[indices]

# Sentence-BERTモデルのロード
model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")

# ベクトル化
embeddings = model.encode(selected_contents.tolist())

# t-SNEによる次元削減
tsne = TSNE(n_components=2, random_state=0, perplexity=min(5, len(selected_contents)-1))
embeddings_2d = tsne.fit_transform(embeddings)

# 可視化（フォントサイズの調整）
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
x_center = np.mean(embeddings_2d[:, 0])
for i, title in enumerate(selected_titles):
    ha = 'right' if embeddings_2d[i, 0] > x_center else 'left'
    plt.annotate(title, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=12, ha=ha)
plt.tight_layout()
plt.show()