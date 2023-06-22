import sqlite3
import os
from tqdm import tqdm
from read_csv import read_csv_files, read_csv_info, data_output
from database import Db
from del_tag import delete_tag


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


def create_only_problem(data_info):
    data_set = []
    dir_path = "./syntax-analysis/Project_CodeNet"
    dir_name = "problem_descriptions"

    for info in tqdm(data_info, postfix="問題のデータセットを作成中"):
        problem_id, dataset = info[0], info[2]
        file_name = problem_id + ".html"
        content = read_file(dir_path, dir_name, file_name)
        content = delete_tag(content)
        if content != None:
            data_set.append([problem_id, dataset, content])

    return data_set     

def create_only_program(problem_id):
    data_set = []
    dir_path = f"./syntax-analysis/project_code_net_python_all/Project_CodeNet/data"

    for id in tqdm(problem_id, postfix="回答のデータセットを生成中"):
        problem_info = read_csv_info(f".\syntax-analysis\Project_CodeNet\metadata\{id}.csv", 1000000)
        for info in problem_info:
            submission_id, problem_id, language, status, cpu_time, memory, code_size = info[0], info[1], info[5], info[7], info[8], info[9], info[10]
            if language == "Python3" or language == "Python":
                dir_name = problem_id
                file_name = submission_id + ".py"
                answer = read_file(dir_path, f"{dir_name}/Python/Project_CodeNet/data/{dir_name}/Python", file_name)
                if answer != None:
                    data_set.append([problem_id, answer, status, cpu_time, memory, code_size])

    return data_set

def main():
    # 1. 問題文のデータセット作成
    data_info = read_csv_info(".\syntax-analysis\Project_CodeNet\metadata\problem_list.csv", 5000)
    # print(data_info)

    # 中身：[id, dataset, problem]
    data_set = create_only_problem(data_info)

    columns = [
        ["problem_id", "STRING"],
        ["dataset", "STRING"],
        ["problem", "TEXT"],
    ]
    dbname = './syntax-analysis/db/mydatasets_all.db'
    table_name = "problems"

    create_db(data_set, columns, dbname, table_name)

    # 2. 解答群のデータセット作成
    problem_id = [data[0] for data in data_info]
    columns = [
        ["problem_id", "STRING"],
        ["program", "TEXT"],
        ["status", "STRING"],
        ["cpu_time", "INT"],
        ["memory", "INT"],
        ["code_size", "INT"],
    ]
    
    # データベースの依存関係を追加
    relation = [ # new_column, ref_table, ref_column
        ["problem_id", "Problems", "problem_id"],
        ]
    
    # 中身：[id, dataset, problem]
    data_set = create_only_program(problem_id)
    
    dbname = './syntax-analysis/db/mydatasets_all.db'
    table_name = "programs"

    create_db(data_set, columns, dbname, table_name, relation)

def check():
    dbname = './syntax-analysis/db/mydatasets_all.db'
    db = Db(dbname)
    data = db.db_output()
    print(data[0])
    print(len(data))

if __name__ == "__main__":
    # データセット作成
    main()
    
    # データセット確認
    # check()


