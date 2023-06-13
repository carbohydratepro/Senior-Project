# Senior-Project

## Description

自然言語処理や深層学習などを用いて競技プログラミングの問題と回答プログラム間の特徴量を抽出し、問題文が与えられたときに正しい解答プログラムを選択できるようにするAIを構築するというプロジェクトです。

## Directory Structure

以下にこのプロジェクトのディレクトリ構造とそれぞれのディレクトリやファイルの概要を示します。

```
project_name/
│
├── ast-label/ # 
│ ├── output_result/ # 
│ ├── create_bert_model.py # 
│ ├── dcreate_data.py # IBMのProject CodeNetからデータセットを作成するプログラム
│ └── file2.py # file2.pyの説明
│
├── dir2/ # dir2の説明
│ ├── file3.py # file3.pyの説明
│ └── file4.py # file4.pyの説明
│
├── dir3/ # dir3の説明
│ ├── subdir1/ # subdir1の説明
│ │ └── file5.py # file5.pyの説明
│ │
│ └── subdir2/ # subdir2の説明
│ └── file6.py # file6.pyの説明
│
└── README.md # README.mdの説明
```


## File Description

### collect_atcoder.py
```text
AtCoderから問題と解答を収集するプログラム
```

### api_test.py
```text
AtCoderのAPIレスポンスやHTML解析のテストを行う用のプログラム
```

### convert_to_python3.py
```text
指定したディレクトリのpython2で書かれたプログラムをpython3に変換するプログラム
```

### Code-LMs
```text
polycoderをクローンしていろいろいじってる
```

### deepcoder-master
```text
deepcoderをクローンしていろいろいじってる
```

## Installation

```
git clone https://github.com/carbohydratepro/senior-project
pip install requirements.txt
```

## Version
```
OS：windows11
python3.10
pip 23.1.2
```

## Usage

プログラムは独立して動くものもあれば、ほかのファイルに依存して動くものもあります。詳しい内容はDirectory Structureを参考にしてください。

## License

：＞

## Contact

：＞
