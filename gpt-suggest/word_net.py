import nltk
from nltk.corpus import wordnet as wn

def related_terms(keyword, relation_type):
    data = set()
    
    for x in wn.synsets(keyword, lang="jpn"):
        if relation_type == "Synonyms":
            data = data.union(set(x.lemma_names(lang='jpn')))
        
        elif relation_type == "Antonyms":
            for lemma in x.lemmas(lang="jpn"):
                antonyms = lemma.antonyms()
                for antonym in antonyms:
                    data = data.union(set(antonym.name()))
        
        elif relation_type == "Hypernyms":
            hypernyms = x.hypernyms()
            for hypernym in hypernyms:
                data = data.union(set(hypernym.lemma_names(lang='jpn')))
        
        elif relation_type == "Hyponyms":
            hyponyms = x.hyponyms()
            for hyponym in hyponyms:
                data = data.union(set(hyponym.lemma_names(lang='jpn')))
                
        elif relation_type == "Holonyms":
            holonyms = x.member_holonyms() + x.part_holonyms() + x.substance_holonyms()
            for holonym in holonyms:
                data = data.union(set(holonym.lemma_names(lang='jpn')))
        
        elif relation_type == "Meronyms":
            meronyms = x.member_meronyms() + x.part_meronyms() + x.substance_meronyms()
            for meronym in meronyms:
                data = data.union(set(meronym.lemma_names(lang='jpn')))
        
        # 他の関係性も追加できますが、基本的なもののみをここに示しています。
    
    return data

if __name__ == "__main__":
    keyword = "人工知能"
    relation = "Synonyms"
    output = related_terms(keyword, relation)
    print(output)
    
    relation = "Hypernyms"
    output = related_terms(keyword, relation)
    print(output)

    relation = "Hyponyms"
    output = related_terms(keyword, relation)
    print(output)

# 同義語（Synonyms）:

# 同じ意味を持つ単語やフレーズのセット。
# 例：car, auto, automobile, machine, motorcar など。
# 反義語（Antonyms）:

# 対照的な意味を持つ単語。
# 例：hot と cold、big と small など。
# 上位語（Hypernyms）:

# 一般的な意味を持つ単語。既に説明しました。
# 例：bird は sparrow の上位語。
# 下位語（Hyponyms）:

# より具体的な意味を持つ単語。既に説明しました。
# 例：sparrow は bird の下位語。
# 全体語（Holonyms）:

# あるものの全体を示す単語。
# 例：tree は bark の全体語。
# 部分語（Meronyms）:

# あるものの部分を示す単語。
# 例：bark は tree の部分語。
# 属性（Attributes）:

# ある名詞と関連する形容詞。または、ある形容詞と関連する名詞。
# 例：weight と heavy、color と red など。
# 原因語（Entailments）:

# ある動詞が別の動詞の実行を含意する関係。
# 例：to snore は to sleep を含意する。
# 派生語（Derived Forms）:

# ある単語から派生した別の単語。
# 例：electricity と electric、runner と run など。
# トピックドメイン（Domain Topics）:

# ある単語が属するドメインやトピック。
# 例：astronomy は star や planet のドメイン。
