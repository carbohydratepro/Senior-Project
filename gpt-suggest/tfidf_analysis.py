from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
from gdfd import gdfd
from collections import Counter
from operator import itemgetter


# ストップワードの定義
stop_words = ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'も', 'する',
              'から', 'な', 'こと', 'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その',
              'あっ', 'よう', 'また', 'もの', 'という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か', 'だ', 'これ',
              'によって', 'により', 'おり', 'より', 'による', 'ず', 'なり', 'られる', 'において', 'ば', 'なかっ',
              'なく', 'しかし', 'について', 'せ', 'だっ', 'その後', 'できる', 'それ', 'う', 'ので', 'なお', 'のみ',
              'でき', 'れ', 'における', 'および', 'いう', 'さらに', 'でも', 'ら', 'たり', 'その他', 'に関する', 'たち',
              'ます', 'ん', 'なら', 'に対して', '特に', 'せる', '及び', 'これら', 'とき', 'では', 'にて', 'ほか', 
              'ながら', 'うち', 'そして', 'とともに', 'ただし', 'かつて', 'それぞれ', 'または', 'に対する', 'ほとんど',
              'と共に', 'といった', 'です', 'とも', 'ところ', 'ここ',
              '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '10',
              '１', '２', '３', '４', '５', '６', '７', '８', '９', '１０',
              '.', '．', '，', ':', '(', ')', '、', '。', '・', '%', '『', '』', '\n','\n\n', ' ', '　', '」', '「',
              '(cid:15)', '\\', ]

def tfidf(documents, n=20):
    # TF-IDF上位の単語を格納する配列
    keywords = []
    
    # Janomeのトークナイザーをインスタンス化
    t = Tokenizer(wakati=True)

    # TF-IDFベクトルライザーのインスタンス化
    vectorizer = TfidfVectorizer(tokenizer=t.tokenize, stop_words=stop_words, use_idf=True, smooth_idf=True)

    # 文書のベクトル化
    X = vectorizer.fit_transform(documents)

    # 文書ごとに単語とそのTF-IDFスコアを表示
    for i, doc in enumerate(documents):
        # 単語とそのTF-IDFスコアを格納する辞書を作成
        word2tfidf = {word: tfidf for word, tfidf in zip(vectorizer.get_feature_names_out(), X[i].toarray()[0])}
        # スコアで降順にソート
        sorted_word2tfidf = sorted(word2tfidf.items(), key=lambda x: x[1], reverse=True)
        # 上位n単語とそのスコアを表示
        keywords.append(keyword[0] for keyword in sorted_word2tfidf[:n])
        
    return keywords


def count_words(array_2d, n=5):
    # 全ての単語をフラットなリストにする
    flat_list = [word for sublist in array_2d for word in sublist]

    # 単語の頻度をカウント
    word_counter = Counter(flat_list)

    # 単語とその頻度をタプルのリストとして取得し、頻度でソート
    sorted_word_counts = sorted(word_counter.items(), key=itemgetter(1), reverse=True)

    # 頻度がn以上の単語を格納するリスト
    words_over_n = [word for word, count in word_counter.items() if count >= n]

    return sorted_word_counts, words_over_n



def main():
    over = 5
    dbname = './gpt-suggest/db/tuboroxn.db'
    documents = gdfd(dbname, 300)
    documents = [document[-1] for document in documents]
    
    keywords = tfidf(documents, 20)
    sorted_word_counts, words_over_n = count_words(keywords, over)
    
    # 上位の単語とその頻度を表示
    for word, count in sorted_word_counts:
        print(f"Word: {word}, Count: {count}")

    # 頻度がn以上の単語を表示
    print(f"Words with frequency over {over}: {words_over_n}")
    
    
    
    
if __name__ == "__main__":
    main()