import spacy
from spacy import displacy

# GiNZAをロード
nlp = spacy.load("ja_ginza")

# 解析するテキスト (例: 論文のタイトルや抄録)
text = "ARを用いた観光情報提示Androidアプリケーション"
# verbにかかっているobjとその周辺単語をMASKする

# テキストを処理
doc = nlp(text)

# 係り受け関係を表示
for token in doc:
    print(f"{token.text} <--{token.dep_}-- {token.head.text}")

# 係り受けツリーを視覚化（オプション）
displacy.serve(doc, style="dep")


# 係り受け関係 (Dependency Relations):

# nsubj: 名詞主語。動詞の主語となる名詞です。
# obj: 目的語。動詞の目的語となる名詞や代名詞です。
# amod: 形容詞修飾語。名詞を修飾する形容詞です。
# advmod: 副詞修飾語。動詞や形容詞、他の副詞を修飾する副詞です。
# root: 係り受け木の根。通常、文の主要な動詞です。
# 他にも多くの係り受け関係タグがあり、それらの完全なリストと説明はUniversal Dependencies documentationで見ることができます。

# 品詞タグ (Part-of-Speech Tags):

# NOUN: 名詞
# VERB: 動詞
# ADJ: 形容詞
# ADV: 副詞
# ADP: 接置詞や助詞（日本語では助詞に相当）
# PRON: 代名詞
# DET: 限定詞
# PUNCT: 句読点
# これらのタグは、単語の文法的なカテゴリを示します。品詞タグの完全なリストと説明はUniversal POS tags documentationで見ることができます。

# 係り受け関係を用いるでもなんでもいいからキーワードを検出する
# その後上下関係

# http://127.0.0.0:8000