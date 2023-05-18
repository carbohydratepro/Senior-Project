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
    with open("./syntax-analysis/del_tag.py", mode='r', encoding='utf-8') as f:
        code = f.read()

    tree = convert_ast(code)

    visitor = CustomVisitor()
    visitor.visit(tree)

    print(f"Function definitions: {visitor.function_defs}") #定義されている関数の数
    print(f"Function calls: {visitor.function_calls}") #関数が呼び出されている回数

    print(ast.dump(tree, indent=2))



if __name__ == "__main__":
    main()