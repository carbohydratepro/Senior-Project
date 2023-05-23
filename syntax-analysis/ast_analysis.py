import ast

def convert_ast(content):
    try:
        tree = ast.parse(content)
        return tree
    except:
        return None

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


def main():
    def analysis(code):
        tree = convert_ast(code)

        visitor = CustomVisitor()
        visitor.visit(tree)

        print(f"Function definitions: {visitor.function_defs}") #定義されている関数の数
        print(f"Function calls: {visitor.function_calls}") #関数が呼び出されている回数

        print(ast.dump(tree, indent=2))
    
    pathes = [".\syntax-analysis\Project_CodeNet_Python800\p00000\s002191454.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s020852510.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s044374417.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s061434217.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s071165995.py",
              ".\syntax-analysis\Project_CodeNet_Python800\p00000\s080762108.py"]
    
    for path in pathes:
        with open(path, mode='r', encoding='utf-8') as f:
            code = f.read()
            print(code)
            analysis(code)







if __name__ == "__main__":
    main()