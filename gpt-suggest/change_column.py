import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv('./gpt-suggest/roxnbuxn_info.csv')

# カラムの順番を変える
df = df[['年度', '学部', '学科', '名前', '学籍番号', '提出日', 'ファイル名', 'タイトル']]

# CSVファイルの書き出し
df.to_csv('./gpt-suggest/roxn.csv', index=False)
