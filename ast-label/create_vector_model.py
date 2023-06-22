import os
import pandas as pd
import ast
from tqdm import tqdm
from langdetect import detect
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from datasets import read_data, create_datasets

##モデルの
class Model():
    def __init__(self, modelname):
        self.modelname=modelname

    # 保存
    def save(self, model):
        model.save(self.modelname)

    # 読み込み
    def read(self):
        return Doc2Vec.load(self.modelname)

    # 削除
    def delete(self):
        pass

# 与えられたprogramを解析してベクトル化したものを返す関数
def ast_and_dfs(program):
    # コードをASTに変換
    def convert_to_ast(content):
        try:
            tree = ast.parse(content)
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
    
    # AST変換
    tree = convert_to_ast(program)

    if tree is None:
        return None
    else:
        feature_vector = ast_to_feature_vector(tree)
        return feature_vector


def create(modelname, datasets):
    created_data = []

    for dataset in tqdm(datasets):
        problem_id = dataset[0]
        problem = dataset[1]
        vector = ast_and_dfs(dataset[2])
        # 特定の言語の問題のセットのみを使用したい場合は以下の記述を追加
        # if detext_language(problem) == "en":
        if vector is None:
            continue
        else:
            created_data.append(TaggedDocument(vector, [problem_id]))

    model = Doc2Vec(created_data,  dm=1, vector_size=300, window=15, alpha=.025, min_alpha=.025, min_count=1, sample=1e-6)
    Model(f"{modelname}.model").save(model)


def ratingAverage(num): #num：配列
    return sum(num)/len(num)

def isFile(file_name):
    return os.path.isfile(file_name)

# textの言語を判定
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "Could not detect language"

def dataVisualization(data, filename, columns):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{filename}.csv")


def main():
    # 初期情報の設定
    data_num = 100000
    filename = './ast-label/model/PE0103'
    label = ["id", "ploblem", "program"]

    # データセットの読み込み
    datasets = read_data(data_num)

    # モデルの作成とデータセットの外部書き込み
    if not isFile(filename):
        create(filename, datasets)
        dataVisualization(datasets, filename, label)

    # モデルの読み込み
    model = Model(f"{filename}.model").read()

    sim = model.dv.most_similar("p00003")
    print(sim)
    print(ratingAverage([s[1] for s in sim]))

def eval():    
    # 初期情報の設定
    data_num = 100000
    filename = './ast-label/model/PE0101'
    label = ["id", "ploblem", "program"]

    # データセットの読み込み
    datasets = read_data(data_num)

    # モデルの読み込み
    model = Model(f"{filename}.model").read()
    
    def vectorCalculate(model, text): #引数からベクトルを求める関数
        vector = model.infer_vector(text)
        return vector
    
    def variousDataEvaluation():
        evaluation_value = {'correct':0, 'incorrect':0}
        eval_results = [] #problem_id、正解数、不正解数、データ数
        for dataset in tqdm(datasets):
            vector = vectorCalculate(model, ast_and_dfs(dataset[2]))
            result = model.dv.most_similar(vector)
            
            if dataset[0] not in [eval_result[0] for eval_result in eval_results]:
                eval_results.append([dataset[0], 0, 0, 0])

            if result[0][0] == dataset[0]:
                evaluation_value['correct'] += 1
                eval_results[[eval_result[0] for eval_result in eval_results].index(dataset[0])][1] += 1
                eval_results[[eval_result[0] for eval_result in eval_results].index(dataset[0])][3] += 1
            else:
                evaluation_value['incorrect'] += 1
                eval_results[[eval_result[0] for eval_result in eval_results].index(dataset[0])][2] += 1
                eval_results[[eval_result[0] for eval_result in eval_results].index(dataset[0])][3] += 1

        for eval_result in eval_results:
            print(eval_result[0], '&', eval_result[3], "&", "{:.2f}".format(eval_result[1]/(eval_result[1]+eval_result[2])*100), r'\verb|%| \\')
            
        print(len(eval_results))

        return evaluation_value['correct'] / (evaluation_value['correct']+evaluation_value['incorrect']) * 100

    rate = variousDataEvaluation()
    print(rate)

if __name__ == "__main__":
    eval()
    