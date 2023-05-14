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

    def db_output(self):
        #データベースから値を抽出
        conn=sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute('SELECT * from propro')
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
