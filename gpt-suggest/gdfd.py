# get data from database
from database import Db
from tqdm import tqdm
import logging
import random

# ログの設定
logging.basicConfig(level=logging.INFO)

def get_data(dbname):
    command = 'SELECT * from theses'
    db = Db(dbname)
    data = db.db_output(command)
    return data

# データセットからランダムに抽出する関数
def select_random_elements(array, n):
    if n > len(array):
        raise ValueError("n is greater than the length of the array.")
    
    random_elements = random.sample(array, n)
    return random_elements
    
def gdfd(dbname, datanum):
    documents = []

    dataset = get_data(dbname)
    for data in tqdm(dataset):
        if len(data[-1]) > 200 and data[-2] != r"卒業論文\n\n論文題目\n\n([\s\S]*?)\n\n":
            documents.append([data[-2], data[-1]]) # .replace("\n", "").replace("・", "")
        else:
            pass
        
    return select_random_elements(documents, datanum)

if __name__ == "__main__":
    print("Please use this program with an external call.")