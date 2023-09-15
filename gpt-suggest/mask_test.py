import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.optim import AdamW
from janome.tokenizer import Tokenizer

main_text = """
最近世界中には、いろいろな犯罪事件がますます増えている。現在注目されている事件では家の侵入事件、盗難事件、殺害事件、火事、駐車場にある車のいたずらなどがある。
"""

def analyze_text(text):
    t = Tokenizer()
    
    # 結果を保存するためのリスト
    tokens_list = []
    tokens_with_pos_dict = {}
    
    for token in t.tokenize(text):
        tokens_list.append(token.surface)
        tokens_with_pos_dict[token.surface] = token.part_of_speech.split(',')[0]  # 品詞の大分類のみを取得
        
    return tokens_list, tokens_with_pos_dict

def main():
    tokens, tokens_with_pos = analyze_text(main_text)
    model_name = "cl-tohoku/bert-large-japanese"
    unmasker = pipeline('fill-mask', model=model_name)

    for text, pos in tokens_with_pos.items():
        if pos == "名詞":
            # textがmain_text中で1回だけ出現するもののみを置き換える
            if main_text.count(text) == 1:
                masked_text = main_text.replace(text, "[MASK]", 1)
                results = unmasker(masked_text)

                # resultsが辞書の場合、リストとして処理
                if isinstance(results, dict):
                    results = [results]

                print(text)
                for result in results:
                    if isinstance(result, dict) and "token_str" in result:
                        token_str = result["token_str"]
                        score = result["score"]
                        print(f"{token_str}:{score:.5f}")

if __name__ == "__main__":
    main()
    


