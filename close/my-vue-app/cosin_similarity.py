from transformers import BertJapaneseTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import sqlite3

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

            # encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
            #                                truncation=True, return_tensors="pt").to(self.device)
            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="max_length", max_length=512,
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)


# データベースに接続
conn = sqlite3.connect('./close/db/tuboroxn.db')
cursor = conn.cursor()

# SQLクエリを実行
query = """
SELECT year, title, overview, faculty, department
FROM theses;
"""
cursor.execute(query)

# 結果を取得
rows = cursor.fetchall()

# 接続を閉じる
conn.close()

titles = [row[1] for row in rows if row[1] is not None]
contents = [row[2] for row in rows if row[2] is not None]

# titleとcontentをペアにしてリストに格納
title_content_pairs = list(zip(titles, contents))
model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calc_sim(input_content):
    # 入力文章のベクトル化
    input_vec = model.encode([input_content])
    content_vecs = model.encode(contents)

    # 類似度計算
    similarities = [cos_sim(input_vec, vec) for vec in content_vecs]

    sim_ave = [] # 10個 10% 25% 50% 100%
    ca_sim = sorted([similarity[0] for similarity in similarities])[::-1]
    sim_ave.append(sum(ca_sim[:10]) / len(ca_sim[:10]))
    sim_ave.append(sum(ca_sim[:len(ca_sim)//10]) / len(ca_sim[:len(ca_sim)//10]))
    sim_ave.append(sum(ca_sim[:len(ca_sim)//4]) / len(ca_sim[:len(ca_sim)//4]))
    sim_ave.append(sum(ca_sim[:len(ca_sim)//2]) / len(ca_sim[:len(ca_sim)//2]))
    sim_ave.append(sum(ca_sim) / len(ca_sim))

    # 上位5つのタイトルと類似度を取得
    top5_indices = np.argsort([similarity[0] for similarity in similarities])[::-1][:5]
    top5_titles = [
        {
            'year': rows[idx][0],
            'title': title_content_pairs[idx][0],
            'summary': title_content_pairs[idx][1],
            'cosineSimilarity': float(similarities[idx][0])}
        for idx in top5_indices
    ]
    print(top5_indices)

    return top5_titles, sim_ave



