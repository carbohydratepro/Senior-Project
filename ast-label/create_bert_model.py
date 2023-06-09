from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from datasets import read_data
import numpy as np
import logging
import torch

# ログの設定
logging.basicConfig(level=logging.INFO)


def eval(datasets, test_data_num=0):
    # GPUが使用可能であるか確認し、使用可能であればGPUを、そうでなければCPUをデバイスとして設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 学習用データ、テストデータの作成
    train_data_num = len(datasets) - test_data_num

    # データセットをプログラムとラベルに分解
    datasets_programs = [dataset[2] for dataset in datasets]
    datasets_labels = [dataset[0] for dataset in datasets]

    # 学習用データの定義
    train_programs = datasets_programs[0:train_data_num]
    train_labels = datasets_labels[0:train_data_num]

    # テストデータの定義
    test_programs = datasets_programs[train_data_num:-1]
    test_labels = datasets_labels[train_data_num:-1]

    # 1. 特徴量の抽出
    # CodeBERTの事前学習モデルとトークナイザーをロード
    logging.info("Loading CodeBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)

    # プログラムから特徴量を抽出（学習用データ）
    logging.info("Extracting features from train programs...")
    train_features = []
    for train_program in tqdm(train_programs):
        inputs = tokenizer(train_program, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        train_features.append(outputs.last_hidden_state[0].mean(0).detach().cpu().numpy())
    train_features = np.array(train_features)
  

    # 2. モデルの学習
    # ランダムフォレスト分類器を学習
    logging.info("Training the classifier...")
    clf = RandomForestClassifier()
    clf.fit(train_features, train_labels)

    # 3. 予測
    # 新しいプログラムから特徴量を抽出し、ラベルを予測
    logging.info("Predicting the label of a new program...")
    test_features = []
    for test_program in tqdm(test_programs):
        inputs = tokenizer(test_program, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        test_features.append(outputs.last_hidden_state[0].mean(0).detach().cpu().numpy().reshape(1, -1))
    test_features = np.array(test_features)


    # 4. 評価
    # 評価結果を格納する配列を定義
    evaluation_value = {'correct':0, 'incorrect':0}
    eval_results = [] #problem_id、正解数、不正解数、データ数
    for test_feature, test_label in tqdm(zip(test_features, test_labels)):
        prediction = clf.predict(test_feature)
        
        if test_label not in [eval_result[0] for eval_result in eval_results]:
            eval_results.append([test_label, 0, 0, 0])

        if prediction == test_label:
            evaluation_value['correct'] += 1
            eval_results[[eval_result[0] for eval_result in eval_results].index(test_label)][1] += 1
            eval_results[[eval_result[0] for eval_result in eval_results].index(test_label)][3] += 1
        else:
            evaluation_value['incorrect'] += 1
            eval_results[[eval_result[0] for eval_result in eval_results].index(test_label)][2] += 1
            eval_results[[eval_result[0] for eval_result in eval_results].index(test_label)][3] += 1

    for eval_result in eval_results:
        print(eval_result[0], ':', eval_result[3], ":", "{:.2f}".format(eval_result[1]/(eval_result[1]+eval_result[2])*100))
        
    print(len(eval_results))

    rate = evaluation_value['correct'] / (evaluation_value['correct']+evaluation_value['incorrect']) * 100
    print(rate)


def main():
    # 初期情報の設定
    data_num = 1000
    test_data_num = int(data_num * 0.3)

    # データセットの読み込み
    datasets = read_data(data_num)
    eval(datasets, test_data_num)


if __name__ == "__main__":
    main()
