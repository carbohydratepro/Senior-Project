import pandas as pd

from transformers import BertModel, BertTokenizer
import torch
import numpy as np

input_title = "拡散モデルを用いた音楽と数学の相関分析"
input_content = """
本論文では、量子コンピューティングの技術を活用して、持続可能な農業システムを設計し、実装するための新しいアプローチを提案します。持続可能な農業は、地球の人口が増加し、気候変動が進行する中で、世界的に重要な課題となっています。この研究は、量子コンピュータの強力な計算能力を利用して、農業生産の効率性と環境への影響を最適化する方法を模索します。

まず、量子コンピューティングが持つ膨大なデータ処理能力を利用して、農業に関わる複雑なデータセットを分析します。これには、気候変動データ、土壌の状態、作物の生育パターンなどが含まれます。量子アルゴリズムを使用して、これらのデータから農業におけるリスクと機会を特定し、より効率的な栽培方法を開発します。

次に、持続可能な農業システムのための量子コンピューティングベースのモデルを構築します。このモデルは、資源の最適な使用、作物の生産性向上、環境への負荷軽減を目指します。量子コンピュータは、異なる農業戦略の結果を高速にシミュレートし、最も持続可能なアプローチを特定します。

さらに、量子コンピューティングを用いて、農業における長期的な気候変動の影響を予測し、適応戦略を提案します。このアプローチにより、農業システムは変化する環境条件に柔軟に対応し、食糧安全保障の向上に貢献します。

最後に、本論文は、量子コンピューティングが持続可能な農業の未来に果たす重要な役割について結論づけます。量子技術を農業に適用することにより、食糧生産の持続可能性を高め、環境への影響を最小限に抑えることが可能になると主張します。これは、地球規模の食糧問題に対する効果的な解決策となる可能性があります。
"""
# CSVファイルのパス
path_titles = './theme-decision-support/data/titles.csv'
path_contents = './theme-decision-support/data/contents.csv'

# CSVファイルを読み込む
# 最初の行（ヘッダー）は無視する
titles = pd.read_csv(path_titles, header=None, skiprows=1)
contents = pd.read_csv(path_contents, header=None, skiprows=1)

# titlesとcontentsの最初のカラムのみを使用
titles = titles[0].tolist()
contents = contents[0].tolist()

# titleとcontentをペアにしてリストに格納
title_content_pairs = list(zip(titles, contents))

title_content_pairs = title_content_pairs[0:1]


# BERTモデルとトークナイザーのロード
model_name = 'cl-tohoku/bert-base-japanese-v3'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 文章をBERTのベクトルに変換する関数
def bert_vectorize(text):
    tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# 類似度を計算する関数
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 入力文章とコンテンツの類似度を計算する関数
def calculate_similarity(input_text, contents):
    input_vec = bert_vectorize(input_text)
    similarities = []
    for content in contents:
        content_vec = bert_vectorize(content)
        sim = cosine_similarity(input_vec, content_vec)
        similarities.append(sim)
    return similarities

# 使用例
# input_text = "ここに入力文章を入れます"
# contents = ["content1", "content2", ...]  # コンテンツのリスト
# title_content_pairs = [("title1", "content1"), ("title2", "content2"), ...]  # タイトルとコンテンツのペア

# 類似度計算
similarities = calculate_similarity(input_content, [content for title, content in title_content_pairs])

print(f"文字数：{len(input_content)}")
print(sum(similarities)/len(similarities))

# 上位5つのタイトルと類似度を取得
top5_indices = np.argsort(similarities)[::-1][:10]
top5_titles = [(title_content_pairs[idx][0], similarities[idx]) for idx in top5_indices]

# 出力
print(top5_titles)

# GiNZAを使用した形態素解析（この環境では実行できない）
# import spacy
# nlp = spacy.load('ja_ginza')
# doc = nlp(input_text)
# for token in doc:
#     print(token.text, token.lemma_, token.pos_)

