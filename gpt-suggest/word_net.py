import nltk
from nltk.corpus import wordnet as wn


def related_words(keyword):

    data = set()
    for x in wn.synsets(keyword, lang="jpn"):
        data = data.union(set(x.lemma_names(lang='jpn')))

    return data

def word_similarity_jp(word1, word2):
    syn = []
    for i in range(len(wn.synsets(word1, lang="jpn"))):
        for j in range(len(wn.synsets(word2, lang="jpn"))):
            syn.append(wn.synsets(word1, lang="jpn")[i].path_similarity(wn.synsets(word2, lang="jpn")[j]))
    return syn

if __name__ == "__main__":
    keyword = "セキュリティ"
    output = related_words(keyword)
    print(output)

    word1="絶対"
    word2="安全"
    print(max(word_similarity_jp(word1, word2)))
