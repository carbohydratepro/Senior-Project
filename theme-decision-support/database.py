import sqlite3

##データベースの
class Db():
    def __init__(self,dbname):
        self.db=dbname

    def db_create(self, command):
        conn=sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute(command)
        conn.commit()
        conn.close()

    def db_input(self, article, sql):
        #値をデータベースに格納
        conn=sqlite3.connect(self.db)
        cur = conn.cursor()
        # sql = 'INSERT INTO propro (problem_id, program) values (?, ?)'
        cur.execute(sql, article)
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
    

def output_datasets():
    dbname = './collect_thesis/db/ieee.db'
    command = "select * from theses"
    db = Db(dbname)
    data = db.db_output(command)
    
    return data