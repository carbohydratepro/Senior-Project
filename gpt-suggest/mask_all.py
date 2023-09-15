import torch
from transformers import BertTokenizer, BertForMaskedLM

def predict_top5(text, mask_idx, model, tokenizer):
    def merge_tokens(tokens):
        merged_tokens = []
        for token in tokens:
            if token.startswith('##'):
                # "##"で始まるトークンは、直前のトークンと結合
                merged_tokens[-1] = merged_tokens[-1] + token[2:]
            else:
                merged_tokens.append(token)
        return merged_tokens
    # テキストをトークン化
    tokens = tokenizer.tokenize(text)
    # tokens = merge_tokens(tokens)
    print(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 指定されたインデックスの単語を保存
    original_token = tokens[mask_idx]
    
    # 指定されたインデックスをMASK
    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    token_ids[mask_idx] = mask_id

    input_ids = torch.tensor([tokenizer.build_inputs_with_special_tokens(token_ids)])
    
    # モデルで予測
    with torch.no_grad():
        output = model(input_ids)

    # トップ5の予測を取得
    predicted_indices = torch.topk(output[0][0, mask_idx], 5).indices.tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)

    return original_token, predicted_tokens

# 事前学習モデルとトークナイザを読み込み
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
model = BertForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_basic_tokenize=True)

text = "私は今から数学を勉強します"
mask_idx = 3
original_token, top5_predictions = predict_top5(text, mask_idx, model, tokenizer)

print(f"Masked Token: {original_token}")
print(f"Top 5 Predictions: {top5_predictions}")

# tokenizerの問題である可能性あり