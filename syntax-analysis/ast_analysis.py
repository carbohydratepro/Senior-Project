import ast

with open("solution.py", "r") as f:
    code = f.read()

tree = ast.parse(code)
