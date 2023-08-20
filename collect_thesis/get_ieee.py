import requests
import json
import os
import random
import time
import glob
from database import Db, check_duplicates, count_year, get_database_size
from tqdm import tqdm
from datetime import datetime
from mail import send_email

#IEEEのapiキー
with open("./collect_thesis/key.txt", "r") as file:
    IEEE_API_KEY = file.read()

# IEEE APIのエンドポイント
base_url = 'http://ieeexploreapi.ieee.org/api/v1/search/articles'

def counter(count_up = True):
    with open("./collect_thesis/count.txt", "r") as file:
        count = int(file.read())
        
    if count_up:
        with open("./collect_thesis/count.txt", "w") as file:
            file.write(str(count+1))
        
    return count
        

def get_thesis():
    err_info = []
    
    count = counter()
    start_record = count * 200 - 199
    
    now = datetime.now()
    today_date = now.strftime("%Y%m%d")
    
    # クエリパラメータを指定してAPIを呼び出す
    params = {
        'apikey': IEEE_API_KEY,
        'format': 'json',
        'max_records': 200,
        'start_record': start_record,
        'sort_order': 'asc',
        'sort_field': 'article_number',
        'start_date': '20000101',
        'end_date': today_date
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        print("API call successful")
    else:
        print("API call unsuccessful with status code:", response.status_code)

    # APIからのレスポンスをJSON形式で取得
    results = json.loads(response.text)
    with open(f'./collect_thesis/json/results_{count}.json', 'w') as file:
        json.dump(results, file, indent=4)

    results = results['articles']
    
    data = []
    for result in tqdm(results):
        try:
            terms = result["index_terms"]["ieee_terms"]["terms"]
            terms = ",".join(map(str, terms))
        except KeyError:
            terms = "None"
        try:
            abstract = result["abstract"]
        except KeyError:
            abstract = "None"
        data.append([result["title"], abstract, result["article_number"], result["publication_year"], result["insert_date"], terms])
            
    columns = [
        ["title", "STRING"],
        ["abstract", "TEXT"],
        ["article_number", "INT"],
        ["publication_year", "INT"],
        ["insert_date", "INT"],
        ["ieee_terms", "TEXT"]
    ]
    
    dbname = './collect_thesis/db/ieee.db'
    table_name = "theses"

    db = Db(dbname)
    db.db_create(table_name, columns)
    command = db.db_create_command(table_name, columns)
    db.db_input(data, command)


def create_mail_text(err, duplicates, years, count, dbsize, jsize):
    mail_text = ""
    
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    mail_text += today_date + "\n\n"
    
    for num, e in err.items():
        mail_text += f"{num}：{e}\n"
    
    mail_text += "\n"
    
    if duplicates:
        mail_text += "重複" + duplicates + "\n\n"
    else:
        mail_text += "重複なし\n\n"
    
    total_bars = 100
    total_years = sum(years.values())

    # 各年号の出現回数を視覚化
    for year, freq in sorted(years.items()):
        bars = "#" * round((freq / total_years) * total_bars)
        mail_text += f"{year} {freq} {bars}\n"
        
    mail_text += "\n"
    
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if dbsize < 1024.0:
            mail_text += f"データベースサイズ：{dbsize:.1f} {unit}\n\n"
            break
        dbsize /= 1024.0
    
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if jsize < 1024.0:
            mail_text += f"バックアップファイルサイズ：{jsize:.1f} {unit}\n\n"
            break
        jsize /= 1024.0
        
    mail_text += "総実行回数：" + str(count) + "\n\n"
        
    return mail_text

def main():
    # 実行状況を格納する辞書
    run_info = {}
    
    # メイン処理
    err_info = {}
    for i in range(1):
        try:
            get_thesis()
        except Exception as e:
            err_info[i+1] = e
            
        time.sleep(1)
    
    dup_info = check_duplicates()
    year_info = count_year()
    
    count = counter(count_up=False)
    db_size = get_database_size()
    
    # 指定されたディレクトリとそのサブディレクトリ内の全ての.jsonファイルを取得
    json_files = glob.glob(os.path.join('./collect_thesis/json/*.json'), recursive=True)
    
    # 各ファイルのサイズを合計
    json_size = sum(os.path.getsize(file) for file in json_files)
    
    mail_text = create_mail_text(err_info, dup_info, year_info, count, db_size, json_size)
    send_email("IEEE論文取得状況", mail_text)
    
if __name__ == "__main__":
    main()