import sqlite3
import os
from collections import Counter

##データベースの
class Db():
    def __init__(self,dbname):
        self.db=dbname

    def db_create(self, table_name, columns):
        # テーブルが存在するかチェック
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cur.fetchone():
            print(f"Table {table_name} already exists.")
            conn.close()
            return

        # テーブルが存在しない場合、新しいテーブルを作成
        columns_str = ', '.join([f"{col[0]} {col[1]}" for col in columns])
        command = (
            f'''CREATE TABLE {table_name}(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_str}
            )'''
        )
        cur.execute(command)
        conn.commit()
        conn.close()


    def db_create_command(self, table_name, columns):
        columns_str = ', '.join([column[0] for column in columns])
        placeholders_str = ', '.join(['?' for _ in columns])
        command = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders_str})"

        return command
    

    def db_input(self, data, command):
        conn=sqlite3.connect(self.db)
        cur = conn.cursor()
        for d in data:
            cur.execute(command, d)
        conn.commit()
        cur.close()
        conn.close()

    def db_output(self, command):
        #データベースから値を抽出
        conn=sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute(command)
        data = cur.fetchall()
        cur.close()
        conn.close()
        return data

    def db_update(self, command):
        conn=sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute(command)

        conn.commit()
        cur.close()
        conn.close()
    
    def db_check_table_exists(self, table_name):
        # 任意のテーブルが存在するかをチェック
        conn=sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")

        # 結果を取得する
        result = cur.fetchone()

        # 接続を閉じる
        cur.close()
        conn.close()

        # テーブルが存在するかどうかを返す
        return result is not None
    


def main():
    data_set = [["year1", "faculty1", "department1", "..."],
                ["year2", "faculty2", "department2", "..."]]
            
    columns = [
        ["year", "INT"],
        ["faculty", "STRING"],
        ["department", "STRING"],
        ["name", "STRING"],
        ["student_id", "STRING"],
        ["submission_date", "STRING"],
        ["file_name", "STRING"],
        ["path", "STRING"],
        ["title", "STRING"],
        ["content", "TEXT"],
    ]
    dbname = './gpt-suggest/db/tuboroxn.db'
    table_name = "theses"

    db = Db(dbname)
    db.db_create(table_name, columns)
    db.db_create_command(table_name, columns)
    db.db_input(data, table_name, columns)

def find_duplicates(arr):
    counts = Counter(arr)
    duplicates = [item for item, count in counts.items() if count > 1]
    return duplicates

def check():
    dbname = './collect_thesis/db/ieee.db'
    command = "select * from theses"
    db = Db(dbname)
    data = db.db_output(command)
    article_numbers = [d for d in data[3]]
    
    duplicates = find_duplicates(article_numbers)
    if duplicates:
        print(f'Duplicates found: {duplicates}')
    else:
        print('No duplicates found.')
    
if __name__ == '__main__':
    check()