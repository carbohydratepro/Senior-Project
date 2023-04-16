import re
import requests
from bs4 import BeautifulSoup

def fetch_solutions(contest_id, task_screen_name):
    contest_url = f"https://atcoder.jp/contests/{contest_id}/standings"
    response = requests.get(contest_url)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data from AtCoder: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    users_data = soup.find_all("tr", class_="standings-result")
    print(users_data)
    exit()

    solutions_data = []
    for user_data in users_data:
        score = user_data.find("td", class_="standings-result-task-cell").find("span", class_="standings-result-score")
        if score:
            user_screen_name = user_data.find("td", class_="standings-result-username").find("a").get_text(strip=True)
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

def find_top_solutions(solutions_data, num_top_solutions):
    solutions_data.sort(key=lambda x: (x["execution_time"] * 0.4 + x["memory_usage"] * 0.4 + x["code_length"] * 0.2))
    return solutions_data[:num_top_solutions]

if __name__ == "__main__":
    contest_id = "abc218"
    task_screen_name = "abc218_a"
    solutions_data = fetch_solutions(contest_id, task_screen_name)
    top_solutions = find_top_solutions(solutions_data, 5)
    print(solutions_data)

    for i, solution in enumerate(top_solutions, 1):
        print(f"=== Top Solution {i} ===")
        print(f"Code Length: {solution['code_length']} B")
        print(f"Execution Time: {solution['execution_time']} ms")
        print(f"Memory Usage: {solution['memory_usage']} KB")
        print(f"Solution Code:\n{solution['solution_code']}\n")

