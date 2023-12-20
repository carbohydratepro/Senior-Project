import pandas as pd
import sqlite3

# CSVファイルを読み込む
titles_df = pd.read_csv('./theme-decision-support/data/titles.csv')
overview_df = pd.read_csv('./theme-decision-support/data/overview.csv')


# インデックスを使って結合する
merged_df = pd.concat([titles_df, overview_df], axis=1)

# データベースに接続
conn = sqlite3.connect('./theme-decision-support/db/tuboroxn.db')
c = conn.cursor()

# 新しいカラムを追加する（例：新しいカラム名は 'new_column' とします）
# c.execute('ALTER TABLE theses ADD COLUMN overview STRING')

# データベースからタイトルのリストを取得
c.execute('SELECT title FROM theses')
db_titles = c.fetchall()

# データベースのタイトルと一致するコンテンツを見つけて、データベースに挿入
for db_title in db_titles:
    match = merged_df[merged_df['title'] == db_title[0]]
    if not match.empty:
        # 一致するコンテンツが見つかった場合、データベースのレコードを更新
        c.execute("UPDATE theses SET overview = ? WHERE title = ?", (match['overview'].iloc[0], db_title[0]))

# 変更をコミットし、接続を閉じる
conn.commit()
conn.close()
