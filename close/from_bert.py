from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance

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


# BERTモデルとトークナイザのロード
model_name = 'cl-tohoku/bert-base-japanese-v3'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 日本語テキストをBERTモデルで埋め込みに変換
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    # 最後の隠れ層の平均を取る
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# テキストのリスト
texts = [input_content, title_content_pairs[0][1]]  # 実際のテキストに置き換えてください

# 埋め込みの取得
embeddings = [get_embedding(text) for text in texts]

# コサイン類似度の計算
cosine_sim = 1 - cosine_distance(embeddings[0].detach().numpy(), embeddings[1].detach().numpy())

print(f"コサイン類似度: {cosine_sim}")