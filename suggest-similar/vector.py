# 必要なライブラリのインストール
# !pip install transformers

import torch
from transformers import BertJapaneseTokenizer, BertModel

# モデルとトークナイザのロード
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_word_embedding(sentence, target_word):
    # トークン化
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # モデルの実行
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 最初のレイヤーの出力を取得
    embeddings = outputs.last_hidden_state[0].numpy()
    
    # target_wordのトークンを取得
    target_tokens = tokenizer.tokenize(target_word)
    if not target_tokens:
        raise ValueError(f"Unable to tokenize word: {target_word}")
    
    # 最初のトークンIDを取得
    token_id = tokenizer.convert_tokens_to_ids(target_tokens[0])
    
    # 最初のトークンの位置を見つける
    idx = (inputs['input_ids'][0] == token_id).nonzero()
    if idx[0].size == 0:
        raise ValueError(f"Token ID for {target_word} ({token_id}) not found in input IDs")
    
    # 対象の単語のベクトルを返す
    return embeddings[idx[0][0]]

# 例の実行
sentence = "私は東京都に住んでいます。"
target_word = "東京都"
embedding = get_word_embedding(sentence, target_word)
print(embedding)
kono