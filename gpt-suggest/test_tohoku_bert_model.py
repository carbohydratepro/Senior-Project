from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np


keyword = "道具"

# 東北大学のBERTモデルとトークナイザーのロード
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

def calculate_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def find_most_similar(word_embedding, word_dict):
    most_similar_word = None
    highest_similarity = -1

    for word, embedding in word_dict.items():
        similarity = calculate_similarity(word_embedding, embedding)
        if similarity > highest_similarity:
            most_similar_word = word
            highest_similarity = similarity

    return most_similar_word, highest_similarity

# 類似語を探すための単語とそのベクトル表現の辞書
word_dict = {}

# 検索対象の単語群
words = ['検知', '動作', '位置', '者', '情報', '学習', '制御', 'カメラ', '台', '遠隔', '動画', '配信', '操作', '家電', '暗号', '化', '鍵', 'アプリケーション', '利用', '起動', 'ツール', 'ローカル', 'プログラム', '履歴', '管理', 'ユーザー', 'ネットワーク', 'ワーク', 'フレーム', 'リモコン', '側', '受信', '送受信', 'html', '文章', '構造', ' 接続', '部', '信号', '周波数', '通信', 'サーバ', 'web', '端末', 'データベース', '科目', 'xml', '要素', '属性', '選択', 'モデル', '検索', '判定', 'ファイル', '関連', '類似', '結果', '抽出', '送信', 'システム', '図', 'コード', 'クライアント', '本', '研究', '閲覧', '編集', '画像', 'ユーザ', '解析', '登録', '音声', '遅延', '秒', 'hz', 'パケット', '描画', 'ページ', '座標', '処理', 'グループ', 'サイト', '単語', '語', 'キーワード', '文字', '使用', 'クラス', 'データ', '入力', '録音', '機能', '画面', '色', '認識', 'idf', 'tf', '検出', '取得', '表示', '実験', '行う', '推薦', '特徴', '度', '感情', 'カテゴリ', '辞書', '距離', '品詞', '件', 'ニュース', 'bluetooth', 'センサ', '加速度', '移動', '投稿', 'android', '再生', '値', 'opencv', 'テキスト', '時間', 'タグ', 'ノード', 'メソッド', 'スレッド', '関係', '参加', '出現', 'パターン', '構築', '下位', '上位', 'ワード', '場合', '名詞', '記事', '屋内', '歩 行', 'iphone', 'ar', '地図', '方位', 'マッチング', 'ツイート', 'twitter', '適合', '手法', '分類', '共有', '商品', '誤差', '率', '判別', '時', '転送', '撮影', '軸', 'm', '方向', 'nanohttpd', '○', '被験者', '回', 'wi', 'fidirect', '地', '音', '識別', '層', '検証', '推定', '関節', 'openpose', '枚', 'cnn']

# 各単語のベクトル表現をBERTモデルを用いて計算
for word in words:
    input_ids = tokenizer.encode(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    word_embeddings = outputs[0][0, 0, :].numpy()  # [CLS]トークンのベクトルを取得
    word_dict[word] = word_embeddings  # ベクトル表現をword_dictに格納

# 「機械学習」のベクトル表現を取得
input_ids = tokenizer.encode(keyword, return_tensors='pt')
with torch.no_grad():
    outputs = model(input_ids)
word_embeddings = outputs[0][0, 0, :].numpy()  # [CLS]トークンのベクトルを取得

most_similar_word, highest_similarity = find_most_similar(word_embeddings, word_dict)

print(f"Most similar word to {keyword} is '{most_similar_word}' with similarity {highest_similarity}.")
