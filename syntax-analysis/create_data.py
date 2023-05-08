import sqlite3
from read_csv import read_csv_files, data_output

def main():
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

if __name__ == "__main__":
    dir_path = "./syntax-analysis/Project_CodeNet/metadata"
    num_files = 10
    num_lines = 10
    data_set = read_csv_files(dir_path, num_files, num_lines)
    data_output(data_set)
