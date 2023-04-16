# AtcoderからABC問題を収集してデータベースに格納するプログラム
import requests
from bs4 import BeautifulSoup
import sqlite3
import re
import time

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

def fetch_solutions(contest_id, task_screen_name):
    contest_url = f"https://atcoder.jp/contests/{contest_id}/standings/json"
    response = requests.get(contest_url)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data from AtCoder API: {response.status_code}")

    try:
        standings_data = response.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON data: {e}")

    task_score = None
    for task in standings_data["TaskInfo"]:
        if task["TaskScreenName"] == task_screen_name:
            task_score = task["Score"]
            break

    if task_score is None:
        raise ValueError("Task not found in the contest")

    top_solutions = []
    for user in standings_data["StandingsData"]:
        if user["TaskResults"][task_screen_name]["Score"] == task_score:
            top_solutions.append(user["UserScreenName"])

    solutions_data = []
    for user_screen_name in top_solutions[:5]:
        code_url = f"https://atcoder.jp/contests/{contest_id}/submissions?f.Task={task_screen_name}&f.User={user_screen_name}&f.Status=AC&f.Language=Python"
        response = requests.get(code_url)
        soup = BeautifulSoup(response.text, "html.parser")
        submission = soup.find("a", text=re.compile("Submission"))
        if submission:
            submission_id = re.search(r"\d+", submission["href"]).group(0)
            code_url = f"https://atcoder.jp/contests/{contest_id}/submissions/{submission_id}"
            response = requests.get(code_url)
            soup = BeautifulSoup(response.text, "html.parser")
            code = soup.find("pre", id="submission-code").get_text(strip=True)

            exec_time = float(soup.find("td", class_="text-right", text=re.compile(r"ms")).get_text(strip=True).rstrip("ms"))
            memory = int(soup.find("td", class_="text-right", text=re.compile(r"KB")).get_text(strip=True).rstrip("KB"))
            code_length = int(soup.find("td", class_="text-right", text=re.compile(r"B")).get_text(strip=True).rstrip("B"))

            solutions_data.append({
                "solution_code": code,
                "code_length": code_length,
                "execution_time": exec_time,
                "memory_usage": memory,
            })

    return solutions_data

def find_best_solution(solutions_data):
    best_solution = None
    best_score = float('inf')

    for solution in solutions_data:
        score = solution["execution_time"] * 0.4 + solution["memory_usage"] * 0.4 + solution["code_length"] * 0.2
        if score < best_score:
            best_score = score
            best_solution = solution

    return best_solution

def main():
    # データベースを作成
    conn = create_database()
    cursor = conn.cursor()
    # ABC問題のURL（例）
    # main関数の引数にURLを指定して、problem_urlを書き換えることで自動化可能
    problem_url = "https://atcoder.jp/contests/abc218/tasks/abc218_a"

    # 任意のcontest_idとtask_screen_nameを取得
    contest_id, task_screen_name = re.search(r"contests/(\w+)/tasks/(\w+)", problem_url).groups()

    # 問題データを取得
    problem_data = fetch_problem_data(problem_url)

    # 解答データを取得
    solutions_data = fetch_solutions(contest_id, task_screen_name)
    best_solution = find_best_solution(solutions_data)

    if not best_solution:
        print("No Python solutions found.")
    else:
        print("=== Best Solution ===")
        print(f"Code Length: {best_solution['code_length']} B")
        print(f"Execution Time: {best_solution['execution_time']} ms")
        print(f"Memory Usage: {best_solution['memory_usage']} KB")
        print(f"Solution Code:\n{best_solution['solution_code']}\n")

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
                ) VALUES (?, ?, ?, ?, ?)
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

