import sqlite3
import ast
from langdetect import detect
from tqdm import tqdm
import random
from transformers import BertTokenizer


def read_data(data_num=100): 
    # データセットからランダムに抽出する関数
    def select_random_elements(array, n):
        if n > len(array):
            raise ValueError("n is greater than the length of the array.")
        
        random_elements = random.sample(array, n)
        return random_elements

    # データベースに接続
    conn = sqlite3.connect(".\syntax-analysis\db\mydatasets.db")

    # カーソルオブジェクトを作成
    cur = conn.cursor()

    # SQLクエリを実行
    print("データベースからデータを読み込み中")
    cur.execute('''
        SELECT problems.problem_id, problems.problem, programs.program
        FROM problems
        INNER JOIN programs ON problems.problem_id = programs.problem_id
    ''')

    data = cur.fetchall() #datas=[[id, problem, program], [id, problem, program], [id, problem, program], ...]

    # 接続を閉じる
    conn.close()

    return select_random_elements(data, data_num)


def create_datasets(data):
    # コードをASTに変換
    def code_to_ast(code):
        try:
            tree = ast.parse(code)
            return tree
        except Exception as e:
            print("Error in parsing:", e)
            return None

    # ASTから特徴ベクトルへ変換
    def ast_to_feature_vector(ast_tree, max_length):
        # DFSを使用してASTツリーを訪問し、ノードタイプ名を収集
        nodes = list(ast.walk(ast_tree))
        feature_vector = [type(n).__name__ for n in nodes]

        # 特徴ベクトルの長さがmax_lengthを超えている場合は切り捨て
        if len(feature_vector) > max_length:
            feature_vector = feature_vector[:max_length]
        # 特徴ベクトルの長さがmax_lengthに満たない場合はパディング
        else:
            feature_vector += ['<PAD>'] * (max_length - len(feature_vector))

        return feature_vector

    # textの言語を判定
    def detect_language(text):
        try:
            lang = detect(text)
            return lang
        except:
            return "Could not detect language"



    datasets = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, data in enumerate(tqdm(data, postfix="データセットをロード中")):
        if detect_language(data[1]) == "en":
            problem_encoding = tokenizer.encode_plus(
                data[1],
                truncation=True,
                padding='max_length',
                max_length=512
                )
                    
            ast_tree = code_to_ast(data[2])
            if ast_tree != None:
                feature_vector = ast_to_feature_vector(ast_tree, max_length=512)
                program_encoding = tokenizer.encode_plus(
                    feature_vector,
                    truncation=True,
                    padding='max_length',
                    max_length=512
                    )
                datasets.append([problem_encoding, program_encoding])
            else:
                continue
    
    return datasets