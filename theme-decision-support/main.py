from transformers import AutoTokenizer, AutoModel, BertJapaneseTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
from database import Db
from tqdm import tqdm
from read_csv import read_csv_files, read_csv_info, data_output
import logging
import json
import os
import pandas as pd

# ログの設定
logging.basicConfig(level=logging.INFO)

# 東北大学が開発した日本語BERTモデルとトークナイザーのロード
# tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# model = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# 東北大学のBERTモデルとトークナイザーのロード
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v3')

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

def save_dict_to_file(dictionary, file_path):
    # Tensorをリストに変換
    converted_dict = {key: value.cpu().numpy().tolist() for key, value in dictionary.items()}
    
    with open(file_path, 'w') as f:
        json.dump(converted_dict, f)
    
def load_dict_from_file(file_path):
    with open(file_path, 'r') as f:
        dictionary = json.load(f)
        # リストをTensorに変換
        converted_dict = {key: torch.tensor(value) for key, value in dictionary.items()}
        
    return converted_dict

def get_data(dbname):
    command = 'SELECT * from theses'
    db = Db(dbname)
    data = db.db_output(command)
    return data

def main():
    logging.info("During data acquisition and preprocessing...")
    # データの準備
    new_title = "量子コンピューティングにおける新たなアルゴリズム設計法とその応用"
    new_content = \
    """
    この論文では、量子コンピューティングにおける新しいアルゴリズム設計法を提案します。具体的には、従来の量子アルゴリズムが解決できる問題領域を広げるため、エンタングルメントとスーパーポジションを更に効果的に活用する新手法を開発しました。この新手法は、特に量子暗号、量子機械学習、量子最適化などの応用分野において優れた結果を示します。さらに、このアルゴリズム設計法を用いた新たなハイブリッド量子古典アルゴリズムを提案し、その計算効率とスケーラビリティについて評価します。最後に、この新しいアルゴリズム設計法が、既存の量子コンピューティング技術の限界を超えて、より広範で複雑な問題に対応できる可能性を示します。本論文は、量子コンピューティングのアルゴリズム開発とその応用分野への深い理解を促進することを目指しています。
    """
    
    # dbname = './gpt-suggest/db/tuboroxn.db'
    # data = get_data(dbname)
    
    # titles = [d[-2] for d in data]
    # contents = [d[-1] for d in data]
    
    titles = pd.read_csv("./theme-decision-support/data/titles.csv").iloc[:,0].tolist()
    contents = pd.read_csv("./theme-decision-support/data/contents.csv").iloc[:,0].tolist()
    details = pd.read_csv("./theme-decision-support/data/details.csv").iloc[:,0].tolist()
    
    new_title = "暗号化による安全な電子メール通信の強化：次世代暗号アルゴリズムの提案と評価"
    new_content = "本論文では、インターネット上でのプライバシーとセキュリティを保証するために必要な、安全な電子メール通信を提供する新たな暗号化アルゴリズムの開発について報告します。初めに、現行の電子メール通信における暗号化手法の課題を明らかにし、その解決策として新しい暗号アルゴリズムの必要性を述べます。次に、我々が開発した次世代暗号アルゴリズムの設計原理と実装手法について詳細に説明します。そして、実際の電子メール通信におけるパフォーマンスとセキュリティのレベルを評価するための一連の実験結果を提示します。これらの結果は、我々の暗号アルゴリズムが既存の方法に比べて優れた安全性を提供し、かつ効率的なパフォーマンスを達成できることを示しています。"
    
    title_content = dict(zip(contents, titles))
    title_detail = dict(zip(titles, details))
    
    # 計算
    new_title_vector = sentence_to_vector(new_title)
    new_content_vector = sentence_to_vector(new_content)

    title_vectors_file_path = "./theme-decision-support/vectors/title_vectors.json"
    content_vectors_file_path = "./theme-decision-support/vectors/content_vectors.json"
    
    # タイトルのベクトルを計算して辞書に格納
    if os.path.exists(title_vectors_file_path):
        title_vectors = load_dict_from_file(title_vectors_file_path)
    else:
        title_vectors = {}
        logging.info("vectorizing all titles...")
        for title in tqdm(titles):
            title_vectors[title] = sentence_to_vector(title)
        save_dict_to_file(title_vectors, title_vectors_file_path)
        
    # 内容のベクトルを計算して辞書に格納
    if os.path.exists(content_vectors_file_path):
        content_vectors = load_dict_from_file(content_vectors_file_path)
    else:
        content_vectors = {}
        logging.info("vectorizing all contents...")
        for content in tqdm(contents):
            content_vectors[content] = sentence_to_vector(content)
        save_dict_to_file(content_vectors, content_vectors_file_path)
        
    # 比較計算
    title_similarities = {}
    content_similarities = {}
    
    logging.info("Calculating similarity between titles...")
    for key, value in tqdm(title_vectors.items()):
        title_similarities[key] = compute_similarity(new_title_vector, value)
        title_similarities = dict(sorted(title_similarities.items(), key=lambda item: item[1], reverse=True))
        
    logging.info("Calculating similarity between contents...")
    for key, value in tqdm(content_vectors.items()):
        content_similarities[key] = compute_similarity(new_content_vector, value)
        content_similarities = dict(sorted(content_similarities.items(), key=lambda item: item[1], reverse=True))
        
    # 結果を表示
    print(f"title:{new_title}")
    # 上位5つのkeyとvalueを表示
    for i, (key, value) in enumerate(title_similarities.items()):
        if i >= 8:
            break
        print(f"  Rank {i+1}: {key} - {value}")
    

    print(f"content:{new_title}")
    # 上位5つのkeyとvalueを表示
    for i, (key, value) in enumerate(content_similarities.items()):
        if i >= 8:
            break
        print(f"  Rank {i+1}: {title_content[key]} - {value}")
        
    print("\n")
    for title in titles:
        print(title, "\n   └", title_detail[title])
        
if __name__ == "__main__":
    main()
