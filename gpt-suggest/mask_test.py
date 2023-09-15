import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.optim import AdamW
from janome.tokenizer import Tokenizer


from janome.tokenizer import Tokenizer

def analyze_text(text):
    t = Tokenizer()
    
    # 結果を保存するためのリスト
    tokens_list = []
    tokens_with_pos_dict = {}
    
    for token in t.tokenize(text):
        tokens_list.append(token.surface)
        tokens_with_pos_dict[token.surface] = token.part_of_speech.split(',')[0]  # 品詞の大分類のみを取得
        
    return tokens_list, tokens_with_pos_dict

if __name__ == "__main__":
    sample_text = "今日はとてもいい天気です。"
    tokens, tokens_with_pos = analyze_text(sample_text)
    
    print("Tokens Only:")
    print(tokens)
    
    print("\nTokens with Part-of-Speech:")
    for text, pos in tokens_with_pos.items():
        print(f"text: {text}, part_of_speech: {pos}")

# model_name = "cl-tohoku/bert-large-japanese"

# unmasker = pipeline('fill-mask', model=model_name)
# print(unmasker("今日の昼食は[MASK]でした。"))

