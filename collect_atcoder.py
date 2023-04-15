# AtcoderからABC問題を収集してデータベースに格納するプログラム
import requests
from bs4 import BeautifulSoup
import sqlite3
import re

# データベースを作成し、テーブルを初期化
def create_database():
    conn = sqlite3.connect("atcoder_data.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_text TEXT,
            solution_code TEXT,
            code_length INTEGER,
            execution_time REAL,
            memory_usage REAL
        )
        """
    )
    conn.commit()
    return conn

# ABC問題のURLからデータを抽出
def fetch_problem_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # 問題文、制約、入力フォーマット、出力フォーマットを抽出
    problem = soup.find("span", {"class": "lang-en"}).parent
    problem_text = problem.find("section").text

    return {
        "problem_text": problem_text,
    }

# 問題の解答を収集
def fetch_solutions(task_screen_name):
    url = f"https://kenkoooo.com/atcoder/resources/accepted-submissions/{task_screen_name}.json"
    response = requests.get(url)
    print(response.text)  # この行を追加
    submissions = response.json()
    exit()

    # 上位5件の正解コードを収集
    top_solutions = sorted(submissions, key=lambda x: (x["length"], x["execution_time"], x["memory"]))[:5]

    solutions_data = []
    for solution in top_solutions:
        # 解答コードを取得
        code_url = f"https://atcoder.jp/contests/{solution['contest_id']}/submissions/{solution['id']}"
        response = requests.get(code_url)
        soup = BeautifulSoup(response.text, "html.parser")
        code = soup.find("pre", id="submission-code").get_text(strip=True)

        solutions_data.append({
            "solution_code": code,
            "code_length": solution["length"],
            "execution_time": solution["execution_time"],
            "memory_usage": solution["memory"],
        })

    return solutions_data


def main():
    # データベースを作成
    conn = create_database()
    cursor = conn.cursor()
    # ABC問題のURL（例）
    problem_url = "https://atcoder.jp/contests/abc218/tasks/abc218_a"

    # 任意のtask_screen_nameを取得
    task_screen_name = re.search(r"tasks/(\w+)", problem_url).group(1)

    # 問題データを取得
    problem_data = fetch_problem_data(problem_url)

    # 解答データを取得
    solutions_data = fetch_solutions(task_screen_name)

    for solution_data in solutions_data:
        # 問題データと解答データを結合
        combined_data = {**problem_data, **solution_data}

        # データベースに保存
        cursor.execute(
            """
            INSERT INTO problems (
                problem_text,
                solution_code,
                code_length,
                execution_time,
                memory_usage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                combined_data["problem_text"],
                combined_data["solution_code"],
                combined_data["code_length"],
                combined_data["execution_time"],
                combined_data["memory_usage"],
            ),
        )

    # コミットして変更を保存
    conn.commit()


if __name__ == "__main__":
    main()

