import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.optim import AdamW

TEXT = """
        カメムシを[MASK]す。
       """

def main():
    model_name = "cl-tohoku/bert-large-japanese"
    unmasker = pipeline('fill-mask', model=model_name)
    
    results = unmasker(TEXT)

    for result in results:
        if isinstance(result, dict) and "token_str" in result:
            token_str = result["token_str"]
            score = result["score"]
            print(f"{token_str}:{score:.5f}")

if __name__ == "__main__":
    main()
    


