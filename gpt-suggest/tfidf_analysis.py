from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
from gdfd import gdfd


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
              '.', '．', '，', ':', '(', ')', '、', '。', '・', '%', '『', '』', '\n',]

def tfidf(documents):
    # Janomeのトークナイザーをインスタンス化
    t = Tokenizer(wakati=True)

    # TF-IDFベクトルライザーのインスタンス化
    vectorizer = TfidfVectorizer(tokenizer=t.tokenize, stop_words=stop_words, use_idf=True, smooth_idf=True)

    # 文書のベクトル化
    X = vectorizer.fit_transform(documents)

    # 文書ごとに単語とそのTF-IDFスコアを表示
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc[:100]}")
        # 単語とそのTF-IDFスコアを格納する辞書を作成
        word2tfidf = {word: tfidf for word, tfidf in zip(vectorizer.get_feature_names_out(), X[i].toarray()[0])}
        # スコアで降順にソート
        sorted_word2tfidf = sorted(word2tfidf.items(), key=lambda x: x[1], reverse=True)
        # 上位10単語とそのスコアを表示
        for word, score in sorted_word2tfidf[:20]:
            print(f"    {word}: {score}")
        print("\n")


def main():
    dbname = './gpt-suggest/db/tuboroxn.db'
    documents = gdfd(dbname, 10)
    documents = [document[-1] for document in documents]
    
    tfidf(documents)
    
    
    
if __name__ == "__main__":
    main()