import ast

# コードをASTに変換
def convert_to_ast(content):
    try:
        tree = ast.parse(content)
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

class CustomVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_defs = 0
        self.function_calls = 0

    def visit_FunctionDef(self, node):
        self.function_defs += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        self.function_calls += 1
        self.generic_visit(node)

def write_to_file(label, data):
    for i, info in enumerate(data, 1):
        with open(f'info{i}.txt', 'w') as f:
            for lbl, item in zip(label, info):
                f.write(f'{lbl}：{item}\n')



def main():
    label = ["ファイル名", "定義されている関数の数", "関数が呼び出されている回数", "DFS", "AST", "プログラム"]
    data = []

    # プログラムのパス
    pathes = [".\syntax-analysis\Project_CodeNet_Python800\p00000\s002191454.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s020852510.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s044374417.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s061434217.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s071165995.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s080762108.py"]
    
    # 与えられた情報を格納してデータを一次元配列で返す関数
    def analysis(code):
        # AST変換
        tree = convert_to_ast(code)

        visitor = CustomVisitor()
        visitor.visit(tree)

        print(f"Function definitions: {visitor.function_defs}") #定義されている関数の数
        print(f"Function calls: {visitor.function_calls}") #関数が呼び出されている回数

        print(ast.dump(tree, indent=2))

    
    for path in pathes:
        with open(path, mode='r', encoding='utf-8') as f:
            code = f.read()
            print(code)
            analysis(code)

    write_to_file(label, data)


if __name__ == "__main__":
    main()