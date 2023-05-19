import sqlite3
import os
from tqdm import tqdm
from read_csv import read_csv_files, read_csv_info, data_output
from database import Db
from del_tag import delete_tag


def isFile(file_name):
    return os.path.isfile(file_name)

def create_db(data, columns, dbname, table_name):
    def create_insert_command(table_name, columns):
        columns_str = ', '.join(columns)
        placeholders_str = ', '.join(['?' for _ in columns])
        command = f"INSERT INTO {table_name} ({columns_str}) values ({placeholders_str})"
        return command
    
    db = Db(dbname)
    # dbが存在しなければ作成
    if not isFile(dbname):
        columns_str = ', '.join([f"{col[0]} {col[1]}" for col in columns])
        command = (
            f'''CREATE TABLE {table_name}(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_str}
            )'''
        )
        db.db_create(command)

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


def create_dataset(data_info):
    data_set = []
    dir_path = "./syntax-analysis/Project_CodeNet_Python800"

    for info in tqdm(data_info, postfix="問題と回答のデータセットを生成中"):
        submission_id, problem_id, language, status = info[0], info[1], info[5], info[7]
        if language == "Python3" and status == "Accepted":
            dir_name = problem_id
            file_name = submission_id + ".py"
            answer = read_file(dir_path, dir_name, file_name)
            if answer != None:
                data_set.append([problem_id, answer])

    return data_set

def create_only_problem(data_info):
    data_set = []
    dir_path = "./syntax-analysis/Project_CodeNet"
    dir_name = "problem_descriptions"

    for info in tqdm(data_info, postfix="問題のデータセットを作成中"):
        problem_id, dataset =info[0], info[2]
        file_name = problem_id + ".html"
        content = read_file(dir_path, dir_name, file_name)
        content = delete_tag(content)
        if content != None:
            data_set.append([problem_id, content, dataset])

    return data_set     

def create_only_program(data_info):
    data_info = read_csv_info(".\syntax-analysis\Project_CodeNet\metadata\problem_list.csv", 100000)
    print(data_info)

    # データセット作成
    data_set = create_only_problem(data_info)

    columns = [
        ["problem_id", "STRING"],
        ["problem", "TEXT"]
        ["FOREIGN KEY(problem_id)", "REFERENCES Problems(problem_id)"]
    ]
    dbname = './syntax-analysis/db/datasets.db'
    table_name = "programs"
    print(data_set)
    create_db(data_set, columns, dbname, table_name)


def main():
    # dir_path = "./syntax-analysis/Project_CodeNet/metadata"
    # num_files = 10
    # num_lines = 1000000
    # data_info = read_csv_files(dir_path, num_files, num_lines)
    # # print(len(data_info))
    # # data_output(data_info)

    # 問題文のデータセット作成
    data_info = read_csv_info(".\syntax-analysis\Project_CodeNet\metadata\problem_list.csv", 100000)
    # print(data_info)

    # データセット作成
    data_set = create_only_problem(data_info)

    columns = [
        ["problem_id", "STRING"],
        ["problem", "TEXT"],
        ["dataset", "STRING"]
    ]
    dbname = './syntax-analysis/db/problems.db'
    table_name = "problems"
    print(data_set)
    exit()
    create_db(data_set, columns, dbname, table_name)

    # 解答群のデータセット作成
    columns = [
        ["problem_id", "STRING"],
        ["problem", "TEXT"],
        ["status", "STRING"],
        ["code_size", "INT"],
        ["FOREIGN KEY(problem_id)", "REFERENCES Problems(problem_id)"]
    ]
    dbname = './syntax-analysis/db/problems.db'
    table_name = "problems"
    print(data_set)
    exit()
    create_db(data_set, columns, dbname, table_name)

def check():
    dbname = './syntax-analysis/db/coding_problems.db'
    db = Db(dbname)
    data = db.db_output()
    print(data[0])

if __name__ == "__main__":
    # データセット作成
    main()
    
    # データセット確認
    # check()


