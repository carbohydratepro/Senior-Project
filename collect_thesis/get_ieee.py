import requests
import json
import os
import random
from database import Db
from tqdm import tqdm
from datetime import datetime

#IEEEのapiキー
with open("./collect_thesis/key.txt", "r") as file:
    IEEE_API_KEY = file.read()

# IEEE APIのエンドポイント
base_url = 'http://ieeexploreapi.ieee.org/api/v1/search/articles'

def counter():
    with open("./collect_thesis/count.txt", "r") as file:
        count = int(file.read())
        
    with open("./collect_thesis/count.txt", "w") as file:
        file.write(str(count+1))
        
    return count
        

def main():
    #queryを用意
    query = 'deep learning'
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
    
    print(len(results))
    
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

if __name__ == "__main__":
    for i in range(4):
        main()