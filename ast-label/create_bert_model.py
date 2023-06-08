from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from datasets import read_data
import numpy as np
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)


def eval(datasets, test_data_num=0):
    # 学習用データ、テストデータの作成
    train_data_num = len(datasets) - test_data_num

    # データセットをプログラムとラベルに分解
    datasets_programs = [dataset[2] for dataset in datasets]
    datasets_labels = [dataset[0] for dataset in datasets]

    # 学習用データの定義
    train_programs = datasets_programs[0:train_data_num]
    trains_labels = datasets_labels[0:train_data_num]

    # テストデータの定義
    test_programs = datasets_programs[train_data_num:-1]
    test_labels = datasets_labels[train_data_num:-1]

    # 1. 特徴量の抽出
    # CodeBERTの事前学習モデルとトークナイザーをロード
    logging.info("Loading CodeBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    # プログラムから特徴量を抽出（学習用データ）
    logging.info("Extracting features from train programs...")
    train_features = []
    for train_program in tqdm(train_programs):
        inputs = tokenizer(train_program, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        train_features.append(outputs.last_hidden_state[0].mean(0).detach().numpy())
    train_features = np.array(train_features)

    # プログラムから特徴量を抽出（テスト用データ）
    logging.info("Extracting features from test programs...")
    test_features = []
    for test_program in tqdm(test_programs):
        inputs = tokenizer(test_program, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        test_features.append(outputs.last_hidden_state[0].mean(0).detach().numpy())
    test_features = np.array(test_features)

    

    # 2. モデルの学習
    # ランダムフォレスト分類器を学習
    logging.info("Training the classifier...")
    clf = RandomForestClassifier()
    clf.fit(features, labels)

    # 3. 予測
    # 新しいプログラムから特徴量を抽出し、ラベルを予測
    logging.info("Predicting the label of a new program...")
    new_program = "new program"
    inputs = tokenizer(new_program, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    new_features = outputs.last_hidden_state[0].mean(0).detach().numpy().reshape(1, -1)

    prediction = clf.predict(new_features)
    print("Predicted label:", prediction)

def main():
    # 初期情報の設定
    data_num = 1000
    test_data_num = 200

    # データセットの読み込み
    datasets = read_data(data_num)
    eval(datasets, test_data_num)


if __name__ == "__main__":
    main()
