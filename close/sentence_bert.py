from transformers import BertJapaneseTokenizer, BertModel
import torch
import numpy as np
import pandas as pd

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


input_title = "OpenPoseを用いた剣道の素振り指導システム"
input_content = """
1990年代，プロ野球で野村克也監督が率いるヤクルトスワローズがID野球というデータを重視した野球を行う事例が出てきている．その影響から近年，スポーツ業界ではITを用いたサービスを利用した選手育成が行われている．実際に，野球ではセンサを利用した，球速やスピンを測定するボールがあり，選手育成のIoT化が進んでいる．そして，最近はコロナウイルスの影響で活動自粛をする必要があり，選手育成のIoT化を進めることで個人練習において選手のデータを解析できれば，活動自粛があってもデータ共有をすることで選手に十分な指導ができると考える．本研究では，背景でも述べたようにIoT化が進んでいないスポーツが多くある中で自分が練習していた剣道の指導するシステムの開発を目的とする．現在サービスとして出ているものは特殊なセンサを用いるものが多いため，本研究ではセンサを用いず，カメラのみで使用できるシステムの開発を目標とする．本研究では，カメラのみで人の骨格情報を取得できるOpenPoseを用いて剣道の基本的な練習方法である素振りに焦点を当てて素振りの指導を行うシステムの開発を行う．システム概要として，youtubeで素振りの動画を上げている剣道6段の方の素振りを解析しそのデータを基にDTW(DynamicTimeWarping)を用いて，DTW距離を求める．DTW距離は値が0に近い程，比較するデータの類似度が高いものである．システム使用前に上級者と初心者の素振りを解析したところ，関節角度の推移のDTW距離は肘8.34，脇7.44となり，初心者は肘47.94，脇27.65となり，初心者の方がDTW距離が大きく，上級者とずれていることが分かった．次に，本実験で作成したシステムを上級者と初心者に使用した時，上級者のシステム使用後の角度の推移のDTW距離は肘6.55，脇6.09となり，改善率は肘は18.9％，脇は14.38％となった．初心者は肘24.14，脇18.98となり，改善率は肘が49.98％，脇が32.02％となった．これらの結果からDTW距離の改善率は初心者の方が高く，初心者に対して有用性があると考える．
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
model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_euclidean_distance(vec1, vec2):
    """
    2つのベクトル間のユークリッド距離を計算する。
    :return: ベクトル間のユークリッド距離
    """
    distance = np.linalg.norm(vec1 - vec2)
    # ユークリッド距離が0の場合、類似度は最大とする
    if distance == 0:
        return np.array([1.0], dtype=np.float32)
    else:
        # 類似度は距離の逆数とする（距離が小さいほど類似度が高い）
        similarity = 1 / (1 + distance)
        return np.array([similarity], dtype=np.float32)

# 入力文章のベクトル化
input_vec = model.encode([input_content])

content_vecs = model.encode(contents)
content_vec_average = sum([content_vec.numpy() for content_vec in content_vecs]) / len(content_vecs)

# # 類似度計算
similarities = [cos_sim(input_vec, vec) for vec in content_vecs]

print(f"文字数：{len(input_content)}")
print(sum(similarities)/len(similarities))
print(cos_sim(input_vec, content_vec_average))

# 上位5つのタイトルと類似度を取得
top5_indices = np.argsort([similarity[0] for similarity in similarities])[::-1][:10]
top5_titles = [(title_content_pairs[idx][0], similarities[idx][0]) for idx in top5_indices]

# 出力
print(top5_titles)

