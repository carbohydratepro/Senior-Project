import sqlite3
import os
from tqdm import tqdm
from read_csv import read_csv_files, read_csv_info, data_output
from database import Db


def isFile(file_name):
    return os.path.isfile(file_name)

def create_db(data, columns):
    dbname = './syntax-analysis/db/coding_problems.db'
    db = Db(dbname)
    # dbが存在しなければ作成
    if not isFile(dbname):
        columns_str = ', '.join([f"{col[0]} {col[1]}" for col in columns])
        command = (
            f'''CREATE TABLE propro(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_str}
            )'''
        )
        db.db_create(command)

    for d in data:
        db.db_input(d)

def read_python_file(dir_path, dir_name, file_name): # dir_path配下にあるdir_nameディレクトリのfile_nameの中身を返す関数
    file_path = dir_path+ "/"+ dir_name + "/" + file_name + ".py"
    try:
        with open(file_path, mode='r', encoding='utf-8') as py_file:
            file_content = py_file.read()
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
            file_name = submission_id
            answer = read_python_file(dir_path, dir_name, file_name)
            if answer != None:
                data_set.append([problem_id, answer])

    return data_set

def create_only_problem(data_info):
    pass       

def main():
    # dir_path = "./syntax-analysis/Project_CodeNet/metadata"
    # num_files = 10
    # num_lines = 1000000
    # data_info = read_csv_files(dir_path, num_files, num_lines)
    # # print(len(data_info))
    # # data_output(data_info)

    data_info = read_csv_info(".\syntax-analysis\Project_CodeNet\metadata\problem_list.csv", 10)
    print(data_info)
    exit()
    # データセット作成
    data_set = create_dataset(data_info)
    
    columns = [
        ["problem", "TEXT"]
    ]
    create_db(data_set, columns)

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


