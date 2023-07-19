from transformers import AutoTokenizer, AutoModel, BertJapaneseTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
from database import Db
from tqdm import tqdm
import logging
import json
import os

# ログの設定
logging.basicConfig(level=logging.INFO)

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
    この論文では、量子コンピューティングにおける新しいアルゴリズム設計法を提案します。具体的には、従来の量子アルゴリズムが解決できる問題領域を広げるため、エンタングルメントとスーパーポジションを更に効果的に活用する新手法を開発しました。

    この新手法は、特に量子暗号、量子機械学習、量子最適化などの応用分野において優れた結果を示します。さらに、このアルゴリズム設計法を用いた新たなハイブリッド量子古典アルゴリズムを提案し、その計算効率とスケーラビリティについて評価します。

    最後に、この新しいアルゴリズム設計法が、既存の量子コンピューティング技術の限界を超えて、より広範で複雑な問題に対応できる可能性を示します。本論文は、量子コンピューティングのアルゴリズム開発とその応用分野への深い理解を促進することを目指しています。
    """
    
    # dbname = './gpt-suggest/db/tuboroxn.db'
    # data = get_data(dbname)
    
    # titles = [d[-2] for d in data]
    # contents = [d[-1] for d in data]
    
    titles = [
        "人工知能による環境保全：ディープラーニングを活用した生物多様性の評価", # 適当
        "量子エラー修正：新たなデコーディングアルゴリズムの開発と評価", # 似ている
        "持続可能な都市開発における緑地利用：心理的健康効果の実証研究", # 異なる
        "ソーシャルメディアがもたらす恋愛観の変化：現代の恋愛傾向に対する影響", # 口調が異なる
        "一日中ネコ動画を見続ける行為における心理的影響", # 論文としては不十分
        "量子アルゴリズムの高速化：効率的なルーチンの設計" # 一つ目と類似しているけれども概要が短いテーマを生成してください
    ]
    contents = [
        """
        この論文では、人工知能（AI）が地球の生物多様性の評価と保全にどのように貢献できるかを探ります。具体的には、ディープラーニングを用いた新しい生物種識別手法を開発し、その効果と可能性を検証します。

        この新しい手法では、ディープラーニングの強力なパターン認識能力を活用し、複数の環境センサーからのデータを用いて各種生物を自動的に識別します。これにより、従来手作業で行われていた分類作業を大幅に高速化し、さらに精度も向上します。

        さらに、この論文では、この技術が生物多様性の評価と保全にどのように応用され得るか、またその利点と課題についても議論します。具体的には、森林、海洋、湿地などの各種生態系における生物種の識別、生物群集の動態の追跡、侵略的外来種の早期検出などについて考察します。

        本論文は、AIの技術が自然環境の保全にどのように貢献できるか、そしてその可能性を広げるためにどのような課題があるかについて、深い理解を提供することを目指します。
        """,
        """
        この論文では、量子エラー修正の新たな手法を提案し、その有効性と可能性について議論します。特に、我々は新しいデコーディングアルゴリズムを開発し、それを用いて量子ビットの誤りをより効率的に修正することが可能になります。

        まず、我々の新たなデコーディングアルゴリズムの詳細について述べます。このアルゴリズムは、エラーが生じた量子ビットの特定と修正を効率的に行うことを目指して設計されています。特に、我々の手法は従来のアルゴリズムに比べて高速であり、また多くのエラーパターンに対してより高い修正成功率を示します。

        次に、この新たなアルゴリズムを用いて、具体的な量子コードに対するエラー修正の性能を評価します。様々なエラーパターンとエラーレートに対してシミュレーションを行い、その結果を報告します。

        この研究は、現実の量子デバイスにおけるエラーの影響を抑制し、高い信頼性を達成するための一歩を提供します。また、本論文は量子エラー修正理論とその応用について新たな視点を提供することを目指しています。
        """,
        """
        本論文では、都市開発における緑地利用が市民の心理的健康にどのように影響するかを探ります。ここでは、緑地の利用が心理的ストレスの緩和、気分の向上、生活満足度の向上等にどの程度貢献するかについて、エビデンスベースの調査を行い、その結果を報告します。

        初めに、都市における緑地の心理的効果に関する先行研究をレビューします。その後、我々が実施した独自の大規模調査の結果を発表します。この調査では、都市住民の緑地利用の頻度、利用形態、そしてそれらが心理的健康指標にどのように関連するかを詳細に分析しました。

        さらに、この研究の結果をもとに、都市開発における緑地設計と管理の推奨事項を提案します。具体的には、都市住民の心理的健康を最大限に高めるために、どのような種類の緑地が必要であり、それらをどのように配置し、管理すべきかについての指針を提供します。

        本論文は、持続可能な都市開発と心理的健康という二つの重要なテーマを結びつけることにより、新たな視点を提供します。また、我々の結果は、都市の計画者や政策立案者がより健康的な都市環境を作り上げるための参考になることを目指しています。
        """,
        """
        この論文では、ソーシャルメディアが現代の恋愛観にどのような影響を与えているかについて調べてみます。言い換えれば、インスタグラムやティックトックなんかが、私たちが恋愛にどう向き合うか、それをどう考えるかに影響を与えているのかを掘り下げていきます。

        まず、我々が実施したアンケート調査の結果を紹介します。これは、ソーシャルメディア利用者たちがどのような恋愛観を持っているのか、またそれが彼らのパートナーに対する期待や関係性の評価にどう影響しているのかを明らかにするためのものです。

        さらに、我々はソーシャルメディア上の恋愛に関する投稿やトレンドについても分析します。これにより、ソーシャルメディアが恋愛観を形成する上でどのような役割を果たしているのかを明らかにします。

        結局のところ、この論文はソーシャルメディアが私たちの恋愛観をどう変えているかについての深い洞察を提供します。それが健全なものなのか、それとも問題があるのか、そして私たちがどう対応すべきかについて考える一助となることを目指しています。
        """,
        """
        本論文では、一日中ネコ動画を見続ける行為が人間の心理状態にどのような影響を及ぼすかについて探ります。とは言っても、科学的な研究として成り立つのか？それはあなたの判断にお任せします。

        初めに、インターネット上で公開されている様々なネコ動画を対象に、それぞれの動画が視聴者に与える可能性のある感情的影響について考察します。さらに、一日中これらの動画を見続ける行為が、心理的なストレスや日常生活の遂行能力、そして全般的な幸福感にどのような影響を及ぼすかについて予測します。

        次に、自発的に一日中ネコ動画を見続けることを選んだ被験者の少数グループを対象に、その心理状態を追跡調査します。この調査結果は、あくまで参考情報として提供され、決して一般化すべきではないということを念頭に置いておくべきです。

        この論文が提供するものは、確かにアカデミックな視点から見れば、論文としての価値が疑問視されるかもしれません。しかし、一日中ネコ動画を見る行為が私たちの心にどのような影響を与えるか、という問いに対する一つの試みであることは間違いありません。
        """,
        """
        この研究では、量子アルゴリズムの効率性を向上させるための新たなルーチンを提案します。我々の方法は、アルゴリズムの計算時間を大幅に短縮し、量子コンピューティングのパフォーマンスを向上させる可能性を秘めています。
        """
    ]
    
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
        if i >= 5:
            break
        print(f"  Rank {i+1}: {key} - {value}")
    

    print(f"content:{new_content}")
    # 上位5つのkeyとvalueを表示
    for i, (key, value) in enumerate(content_similarities.items()):
        if i >= 5:
            break
        print(f"  Rank {i+1}: {key} - {value}")
        
        
if __name__ == "__main__":
    main()
