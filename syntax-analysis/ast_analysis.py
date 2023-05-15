import ast

def convert_ast(content):
    try:
        tree = ast.parse(content)
        return tree
    except:
        return None
        
def main():
    with open("./syntax-analysis/del_tag.py", "r") as f:
        code = f.read()

    tree = convert_ast(code)
    print(tree)

if __name__ == "__main__":
    main()