from transformers import AutoTokenizer, AutoModel, BertJapaneseTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
from database import Db

# 東北大学が開発した日本語BERTモデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# 東北大学のBERTモデルとトークナイザーのロード
# tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-v3')
# model = BertModel.from_pretrained('bert-base-japanese-v3')

# GPUが利用可能であればGPUを、そうでなければCPUを使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルをデバイスに移動
model = model.to(device)

def sentence_to_vector(sentence):
    # 文章をトークン化し、モデルに入力できる形式に変換
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding='max_length').to(device)

    # モデルを使って文章の表現を計算
    with torch.no_grad():
        outputs = model(**inputs)

    # 最初のトークン（[CLS]トークン）の表現を取得
    sentence_embedding = outputs.last_hidden_state[0, 0]

    return sentence_embedding

def compute_similarity(vector1, vector2):
    # コサイン類似度を計算
    similarity = 1 - cosine(vector1.cpu(), vector2.cpu())

    return similarity

def main():
    # データの準備
    new_title = "量子コンピューティングにおける新たなアルゴリズム設計法とその応用"
    new_content = """
    この論文では、量子コンピューティングにおける新しいアルゴリズム設計法を提案します。具体的には、従来の量子アルゴリズムが解決できる問題領域を広げるため、エンタングルメントとスーパーポジションを更に効果的に活用する新手法を開発しました。

この新手法は、特に量子暗号、量子機械学習、量子最適化などの応用分野において優れた結果を示します。さらに、このアルゴリズム設計法を用いた新たなハイブリッド量子古典アルゴリズムを提案し、その計算効率とスケーラビリティについて評価します。

最後に、この新しいアルゴリズム設計法が、既存の量子コンピューティング技術の限界を超えて、より広範で複雑な問題に対応できる可能性を示します。本論文は、量子コンピューティングのアルゴリズム開発とその応用分野への深い理解を促進することを目指しています。
"""
    dbname = './gpt-suggest/db/tuboroxn.db'
    data = get_data(dbname)
    
    only_titles = [d[-2] for d in data]
    only_contents = [d[-1] for d in data]
    
    
    

    

if __name__ == "__main__":
    # 2つの文章を定義
    sentence1 = "私は犬が好きです。"
    sentence2 = "それは機械学習を用いたメガネの学習です。"

    # 類似度を計算
    similarity = compute_similarity(sentence1, sentence2)
    print(f"Similarity: {similarity}")
