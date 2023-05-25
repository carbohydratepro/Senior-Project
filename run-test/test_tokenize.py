from transformers import AutoTokenizer

python_code = r"""def say_hello():
    print("hello, World!")

# Print it
say_hello()
"""
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer(python_code).tokens())