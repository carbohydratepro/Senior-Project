import ast

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

code = """
def fib(n):
    if n <= 1:
       return n
    else:
       return(fib(n-1) + fib(n-2))
"""

ast_tree = code_to_ast(code)
feature_vector = ast_to_feature_vector(ast_tree)
print(feature_vector)
