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
    new_title = "学術情報のオープンアクセスシステムによる大学の卒業研究論文の利用と情報構築"
    new_content = \
    """
    導入
    この部分では、研究の背景と本システムの必要性について説明します。具体的には、卒業論文という重要な知識資源が限られた範囲でしか利用されていない現状と、これを公開・共有することによる学術研究の進展や社会貢献の可能性について述べます。

    システム設計
    このセクションでは、PDFを用いた公開研究情報構築システムの設計について詳細に述べます。このシステムは、卒業論文のPDFをアップロード、検索、ダウンロードが可能となるように設計されています。また、メタデータの管理、ユーザー認証、アクセス制御などの重要な機能も導入します。

    使用技術
    システムの開発には、WebフレームワークとしてPythonのFlask、データベースとしてMySQL、フルテキスト検索エンジンとしてElasticsearchを採用します。また、PDFの解析・表示には、AdobeのPDF.jsなどのライブラリを利用します。

    実装と評価
    本研究では、システムを実装し、研究室の実環境で運用・評価します。具体的には、卒業論文のアップロード数、ダウンロード数、ページビュー数などの指標を用いてシステムの利用状況と有効性を評価します。

    結論
    システムの運用と評価の結果、本システムが卒業論文の公開・利用を促進し、学術研究の進展に寄与することが確認されます。さらなる機能追加や改良の方向性についても考察します。
    """
    
    dbname = './gpt-suggest/db/tuboroxn.db'
    data = get_data(dbname)
    
    titles = [d[-2] for d in data]
    contents = [d[-1] for d in data]
    
    # titles = pd.read_csv("./theme-decision-support/data/titles.csv").iloc[:,0].tolist()
    # contents = pd.read_csv("./theme-decision-support/data/contents.csv").iloc[:,0].tolist()
    details = pd.read_csv("./theme-decision-support/data/details.csv").iloc[:,0].tolist()
    
    print(len(titles), len(contents))
    # new_title = "論文テーマ決め支援ツールの作成"
    # new_content = "本論文の著者が所属する研究室では, 論文のテーマ決めに苦労している人がほとんどである. その理由として, 自身の興味のある分野を研究のテーマとして結び付けるのが難しく, また金銭面や技術面など考慮することがたくさんあり, 自分の考えたテーマが実現可能であるかを精査するのに多くの時間を要している. 研究に費やせる時間は限られている為, 失敗しないテーマ選びというのが非常に重要となってくる. そのため, 本研究では自然言語処理をはじめとする様々な手法を用いて, 研究者が円滑にテーマ決めを行えるようなツールを作成する. "
    
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
    content_similarities_reverse = {}
    
    logging.info("Calculating similarity between titles...")
    for key, value in tqdm(title_vectors.items()):
        title_similarities[key] = compute_similarity(new_title_vector, value)
        title_similarities = dict(sorted(title_similarities.items(), key=lambda item: item[1], reverse=True))
        
    logging.info("Calculating similarity between contents...")
    for key, value in tqdm(content_vectors.items()):
        content_similarities[key] = compute_similarity(new_content_vector, value)
        content_similarities = dict(sorted(content_similarities.items(), key=lambda item: item[1], reverse=True))
        
    for key, value in tqdm(content_vectors.items()):
        content_similarities_reverse[key] = compute_similarity(new_content_vector, value)
        content_similarities_reverse = dict(sorted(content_similarities.items(), key=lambda item: item[1], reverse=False))
        
    # 結果を表示
    # print(f"title:{new_title}")
    # # 上位5つのkeyとvalueを表示
    # for i, (key, value) in enumerate(title_similarities.items()):
    #     if i >= 8:
    #         break
    #     print(f"  Rank {i+1}: {key} - {value}")
        
        
    print(f"content:{new_title}")
    # 上位5つのkeyとvalueを表示
    for i, (key, value) in enumerate(content_similarities.items()):
        if i >= 8:
            break
        print(f"  Rank {i+1}: {title_content[key]} - {value}")
        
    print("\n")
    
    # 下位5つのkeyとvalueを表示
    for i, (key, value) in enumerate(content_similarities_reverse.items()):
        if i >= 8:
            break
        print(f"  Rank {i+1}: {title_content[key]} - {value}")
    # for title in titles:
    #     print(title, "\n   └", title_detail[title])
        
if __name__ == "__main__":
    main()
