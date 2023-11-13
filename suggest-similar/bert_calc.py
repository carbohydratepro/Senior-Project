import torch
from transformers import BertJapaneseTokenizer, BertModel

# モデルとトークナイザーの初期化
model_name = 'cl-tohoku/bert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# テキストをベクトルに変換する関数
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()

# 「男」「王様」「女」の埋め込みを取得
man_embedding = text_to_embedding("男")
king_embedding = text_to_embedding("王様")
woman_embedding = text_to_embedding("女")

# アナロジー: 男 + 王様 - 女 = ?
analogy_embedding = king_embedding - woman_embedding + man_embedding

# コサイン類似度による最も近い単語を見つけるための関数
def find_closest_embeddings(target_embedding, word_list):
    closest_word = None
    max_cosine_similarity = -float('Inf')
    
    for word in word_list:
        word_embedding = text_to_embedding(word)
        cosine_similarity = torch.nn.functional.cosine_similarity(target_embedding, word_embedding)
        if cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = cosine_similarity
            closest_word = word
    
    return closest_word

# 候補のリスト（本来であればもっと多くの適切な候補が必要）
candidates = ["女王",'豆']

# 最も近い単語を見つける
closest_word = find_closest_embeddings(analogy_embedding, candidates)
print("The closest word to the analogy '男 + 王様 - 女' is:", closest_word)
