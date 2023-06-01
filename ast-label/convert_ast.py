import ast

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

    return nodes, feature_vector


def write_to_file(label, data):
    for i, info in enumerate(data, 1):
        with open(f'info{i}.txt', 'w') as f:
            for lbl, item in zip(label, info):
                f.write(f'{lbl}：{item}\n')



def main():
    label = ["ファイル名", "定義されている関数の数", "関数が呼び出されている回数", "DFS", "DFS_vector", "AST", "プログラム"]
    data = []

    # プログラムのパス
    pathes = [".\syntax-analysis\Project_CodeNet_Python800\p00000\s002191454.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s020852510.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s044374417.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s061434217.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s071165995.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s080762108.py"]
    
    # 与えられた情報を格納してデータを一次元配列で返す関数
    def analysis(path, code):
        # AST変換
        tree = convert_to_ast(code)

        # ASTを訪問するオブジェクトを作成
        visitor = CustomVisitor()
        visitor.visit(tree)

        function_definitions = visitor.function_defs
        function_calls = visitor.function_calls

        program_ast = ast.dump(tree, indent=2)

        nodes, feature_vector = ast_to_feature_vector(program_ast)

        return [path, function_definitions, function_calls, nodes, feature_vector, program_ast, code]
    
    for path in pathes:
        with open(path, mode='r', encoding='utf-8') as f:
            code = f.read()
            data.append(analysis(path, code))

    # データをテキストファイルに書き込み
    write_to_file(label, data)



if __name__ == "__main__":
    main()