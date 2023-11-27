from transformers import BertTokenizer, BertForMaskedLM
import torch

# 日本語のBERTモデルとトークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# マスクされた日本語の文章の準備（例: "私は[MASK]を食べます"）
text = """
私は[MASK]を食べます
"""
masked_index = tokenizer.encode(text, add_special_tokens=True).index(tokenizer.mask_token_id)

# 文章をトークンに変換
inputs = tokenizer.encode(text, return_tensors="pt")

# BERTモデルで予測
with torch.no_grad():
    predictions = model(inputs)[0]


words = ["椅子", "リンゴ", "朝食", "夕食", "カバ", "何"]

for word in words:
    # 特定の単語の確率を取得
    predicted_index = tokenizer.convert_tokens_to_ids(word)
    predicted_prob = torch.softmax(predictions[0, masked_index], dim=-1)[predicted_index]

    print(f"「{word}」の確率は {predicted_prob.item()}")
