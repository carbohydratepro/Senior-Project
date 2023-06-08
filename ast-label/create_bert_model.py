from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from datasets import read_data
import numpy as np
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)


def eval(programs, labels):
    # 1. データの収集
    # ラベル付けされたプログラムのデータセットを用意します。
    # ここでは仮のデータを使用します。
    # programs = ["program1", "program2", "program3", "program4", "program5"]
    # labels = ["label1", "label2", "label2", "label1", "label3"]


    # 2. 特徴量の抽出
    # CodeBERTの事前学習モデルとトークナイザーをロードします。
    logging.info("Loading CodeBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    # プログラムから特徴量を抽出します。
    logging.info("Extracting features from programs...")
    features = []
    for program in tqdm(programs):
        inputs = tokenizer(program, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        features.append(outputs.last_hidden_state[0].mean(0).detach().numpy())
    features = np.array(features)

    # 3. モデルの学習
    # ランダムフォレスト分類器を学習します。
    logging.info("Training the classifier...")
    clf = RandomForestClassifier()
    clf.fit(features, labels)

    # 4. 予測
    # 新しいプログラムから特徴量を抽出し、ラベルを予測します。
    logging.info("Predicting the label of a new program...")
    new_program = "new program"
    inputs = tokenizer(new_program, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    new_features = outputs.last_hidden_state[0].mean(0).detach().numpy().reshape(1, -1)

    prediction = clf.predict(new_features)
    print("Predicted label:", prediction)

def main():
    # 初期情報の設定
    data_num = 100
    label = ["id", "ploblem", "program"]

    # データセットの読み込み
    datasets = read_data(data_num)
    programs = [dataset[2] for dataset in datasets]
    labels = [dataset[0] for dataset in datasets]
    eval(programs, labels)


if __name__ == "__main__":
    main()
