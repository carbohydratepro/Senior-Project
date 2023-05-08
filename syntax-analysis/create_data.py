import sqlite3
import os
from read_csv import read_csv_files, data_output



def create_db():
    conn = sqlite3.connect('./db/coding_problems.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS coding_problems
                (id INTEGER PRIMARY KEY, problem_statement TEXT, solution_code TEXT, problem_id INTEGER)''')

    # 問題文と回答コードのデータを挿入（以下は例）
    data = [
        (1, "問題文1", "解答コード1-1", 1),
        (2, "問題文1", "解答コード1-2", 1),
        # ...
        (600, "問題文20", "解答コード20-30", 20),
    ]

    for item in data:
        c.execute("INSERT INTO coding_problems VALUES (?, ?, ?, ?)", item)

    conn.commit()
    conn.close()

def return_file_data(dir_path, dir_name, file_name): # dir_path配下にあるdir_nameディレクトリのfile_nameの中身を返す関数
    directory = dir_path+ "/"+ dir_name + "/" + file_name
    file_names = []
    for entry in os.listdir(directory):
        file_path = os.path.join(directory, entry)
        if os.path.isfile(file_path):
            file_names.append(entry)

    return file_names


def create_dataset(data_info):
    data_set = []
    for info in data_info:
        submission_id, problem_id, language, status = info[0], info[1], info[5], info[7]
        if language == "Python3" and status == "Accepted":
            print(info)

def main():
    dir_path = "./syntax-analysis/Project_CodeNet/metadata"
    num_files = 10
    num_lines = 1000000
    data_info = read_csv_files(dir_path, num_files, num_lines)
    # data_output(data_info)

    # dir_name = "p00000"
    # file_name = "s002191454.py"
    # dir_path = "./syntax-analysis/Project_CodeNet_Python800"
    # return_file_data(dir_path, dir_name, file_name)

    data_set = create_dataset(data_info)


if __name__ == "__main__":
    main()


