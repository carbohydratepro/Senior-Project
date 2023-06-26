import sqlite3
import os
from tqdm import tqdm
from read_csv import read_csv_files, read_csv_info, data_output
from database import Db
from pdf_to_data import remove_spaces, convert_pdf_to_txt


def isFile(file_name):
    return os.path.isfile(file_name)

def create_db(data, columns, dbname, table_name, relation=None):
    # dbコマンドを作成する関数
    def create_insert_command(table_name, columns):
        columns_str = ', '.join(columns)
        placeholders_str = ', '.join(['?' for _ in columns])
        command = f"INSERT INTO {table_name} ({columns_str}) values ({placeholders_str})"
        return command
    
    db = Db(dbname)

    # テーブルが存在しなければ作成
    if not db.db_check_table_exists(table_name):
        columns_str = ', '.join([f"{col[0]} {col[1]}" for col in columns])
        # データベースの依存関係を追加
        if relation != None:
            for rel in relation:
                new_column, ref_table, ref_column = rel
                columns_str += f', FOREIGN KEY({new_column}) REFERENCES {ref_table}({ref_column})'
        command = (
            f'''CREATE TABLE {table_name}(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_str}
            )'''
        )
        db.db_create(command)

    # データベースにデータを格納
    sql = create_insert_command(table_name, [column[0] for column in columns])
    for d in data:
        db.db_input(d, sql)

def read_file(dir_path, dir_name, file_name): # dir_path配下にあるdir_nameディレクトリのfile_nameの中身を返す関数
    file_path = dir_path+ "/"+ dir_name + "/" + file_name
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            file_content = file.read()
            return file_content
    except:
        return None


def main():
    # 論文の情報ファイルを読み込み
    data_info = read_csv_info("./gpt-suggest/roxn.csv", 500)
    # print(data_info)

    data_set = []

    for info in tqdm(data_info):
        year, faculty, department, name, student_id, submission_date, file_name, title = info
        path = f'../卒論PDF/{year}/{file_name}'
        try:
            pdf_text = convert_pdf_to_txt(path)
            pdf_text = remove_spaces(pdf_text)
        except:
            continue
        
        data_set.append([year, faculty, department, name, student_id, submission_date, file_name, path, title, pdf_text])


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

    create_db(data_set, columns, dbname, table_name)


if __name__ == "__main__":
    # データセット作成
    main()
    
    # データセット確認
    # check()


