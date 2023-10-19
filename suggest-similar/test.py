import spacy

def remove_unnecessary_words(text):
    # GiNZAモデルをロード
    nlp = spacy.load("ja_ginza")
    
    # テキストを解析
    doc = nlp(text)

    # 接続詞などの不要な単語を除去
    # ここでは接続詞(CONJ)、接続助詞(ADP)、補助記号(PUNCT)を除去していますが、
    # 必要に応じて除去する品詞のリストを調整することができます。
    unnecessary_pos = ["CONJ", "ADP", "PUNCT"]
    filtered_tokens = [token.text for token in doc if token.pos_ not in unnecessary_pos]

    # フィルタリングされた単語を連結して返す
    return ' '.join(filtered_tokens)

# 例
text = "彼は学校に行くが、私は家にいる。"
result = remove_unnecessary_words(text)
print(result)
