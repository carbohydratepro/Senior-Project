import sqlite3
import ast
from langdetect import detect
from transformers import BertTokenizer
from tqdm import tqdm

# コードをASTに変換
def code_to_ast(code):
    try:
        tree = ast.parse(code)
        return tree
    except Exception as e:
        print("Error in parsing:", e)
        return None

# ASTから特徴ベクトルへ変換
def ast_to_feature_vector(ast_tree):
    # DFSを使用してASTツリーを訪問し、ノードタイプ名を収集
    nodes = list(ast.walk(ast_tree))
    feature_vector = [type(n).__name__ for n in nodes]
    return feature_vector


def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "Could not detect language"


# データベースに接続
conn = sqlite3.connect(".\syntax-analysis\db\mydatasets.db")

# カーソルオブジェクトを作成
cur = conn.cursor()

# SQLクエリを実行
cur.execute('''
    SELECT problems.problem_id, problems.problem, programs.program
    FROM problems
    INNER JOIN programs ON problems.problem_id = programs.problem_id
''')

datas = cur.fetchall()
datasets = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for data in tqdm(datas):
    if detect_language(data[1]) == "en":
        problem_encoding = tokenizer.encode_plus(
            data[1],
            truncation=True,
            padding='max_length',
            max_length=512
            )
                
        ast_tree = code_to_ast(data[2])
        if ast_tree != None:
            feature_vector = ast_to_feature_vector(ast_tree)
            program_encoding = tokenizer(feature_vector, truncation=True, padding=True)
            datasets.append([problem_encoding, program_encoding])
        else:
            continue

print(datasets[1])

  



# 接続を閉じる
conn.close()
