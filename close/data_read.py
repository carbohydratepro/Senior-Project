import pandas as pd

from transformers import BertModel, BertTokenizer
import torch
import numpy as np

# GPUが利用可能かチェックし、利用可能な場合はGPUを使用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_title = "自然言語処理とブロックチェーン技術を活用したセキュアな通信プロトコル"
input_content = """
本論文では、自然言語処理（NLP）とブロックチェーン技術を組み合わせた革新的なセキュア通信プロトコルを提案します。現代のデジタル通信環境において、セキュリティとプライバシーの確保は非常に重要です。この研究では、自然言語の理解と解析に長けた機械学習モデルと、ブロックチェーンに基づく堅固なセキュリティ構造を組み合わせることで、これらの課題に対処します。研究の最初の段階では、自然言語処理を用いて通信内容のセマンティック分析を行い、機密情報の自動検出を可能にします。この分析により、重要なデータが含まれる通信を特定し、高度なセキュリティ対策を適用します。次に、ブロックチェーン技術を活用して、通信の不変性と追跡不可能性を保証します。このプロトコルは、各通信に一意のトランザクションIDを割り当て、ブロックチェーン上に記録します。これにより、データの改ざんや不正アクセスを効果的に防ぐことができます。この論文は、自然言語処理とブロックチェーン技術の融合により、通信セキュリティの新たな可能性を探ります。セキュリティ専門家、データサイエンティスト、および通信技術者にとって、新しいセキュリティ対策の開発に役立つ知見を提供するものです。
"""
# CSVファイルのパス
path_titles = './theme-decision-support/data/titles.csv'
path_contents = './theme-decision-support/data/overview.csv'

# CSVファイルを読み込む
# 最初の行（ヘッダー）は無視する
titles = pd.read_csv(path_titles, header=None, skiprows=1)
contents = pd.read_csv(path_contents, header=None, skiprows=1)

# titlesとcontentsの最初のカラムのみを使用
titles = titles[0].tolist()
contents = contents[0].tolist()

# titleとcontentをペアにしてリストに格納
title_content_pairs = list(zip(titles, contents))

# title_content_pairs = title_content_pairs[0:1]


# BERTモデルとトークナイザーのロード
model_name = 'cl-tohoku/bert-large-japanese-v2'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

# 文章をBERTのベクトルに変換する関数（バッチ処理）
def bert_vectorize(texts):
    tokens = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length', add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def bert_vectorize_max_pooling(texts):
    tokens = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length', add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
    return torch.max(outputs.last_hidden_state, dim=1).values.squeeze()

def bert_vectorize_cls_token(texts):
    tokens = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length', add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
    # [CLS]トークンは最初のトークンなので、インデックス0を選択
    return outputs.last_hidden_state[:, 0, :].squeeze()


# 類似度を計算する関数
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 入力文章のベクトル化
input_vec = bert_vectorize([input_content]).cpu().numpy()

# コンテンツのベクトル化（バッチ処理）
batch_size = 100
content_vecs = []
for i in range(0, len(contents), batch_size):
    batch_contents = contents[i:i+batch_size]
    vecs = bert_vectorize(batch_contents).cpu().numpy()
    content_vecs.extend(vecs)

# 類似度計算
similarities = [cosine_similarity(input_vec, vec) for vec in content_vecs]

print(f"文字数：{len(input_content)}")
print(sum(similarities)/len(similarities))

print(len(similarities))
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

