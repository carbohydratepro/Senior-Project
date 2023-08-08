from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
from database import Db
import random


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
              '０', '１', '２', '３', '４', '５', '６', '７', '８', '９', '１０',
              '.', '．', '，', ':', '(', ')', '、', '。', '・', '%', '『', '』', '\n','\n\n', ' ', '　', '」', '「',
              '(cid:15)', '\\', '>', '</', '<', '/', '％', '="', 'cid', ')(', 'p', '•', '-', ',', '（', '）',
              '"', ');', '}', '{', '_',]


def get_data(dbname):
    command = 'SELECT * from theses'
    db = Db(dbname)
    data = db.db_output(command)
    return data


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
        keywords.append([keyword[0] for keyword in sorted_word2tfidf[:n]])
        
    return keywords


def randomChoice():
    dbname = './gpt-suggest/db/tuboroxn.db'
    data = get_data(dbname)
    
    titles = [d[-2] for d in data]
    contents = [d[-1] for d in data]
    
    index = random.randint(0, len(data))
    
    return titles[index], contents[index]
    
def main():
    new_title = "学術情報のオープンアクセスシステムによる大学の卒業研究論文の利用と情報構築"
    new_content = \
    """
    近年、多くの大学が学術情報をオープンアクセスの形式で公開しています。本論文では、大学の卒業生が提出した研究論文をオープンアクセスシステムで公開し、利用することによる情報構築と共有について研究します。

    論文では、まずオープンアクセスシステムの概念とその利点について解説します。オープンアクセスは、学術情報を無料で公開し、誰もが自由にアクセスできることを目指す取り組みです。その後、大学の卒業生が提出した研究論文の重要性と潜在的な価値について議論します。

    次に、既存のオープンアクセスシステムや大学のリポジトリを活用した情報構築の実践事例を紹介します。学術情報のオープンアクセス化により、研究成果を広く共有できることで、新たな知識の創造や学際的な研究の促進に寄与している例があります。

    さらに、大学の卒業研究論文をオープンアクセスで公開することによる利点と課題を分析します。情報の共有による知識の普及とアクセシビリティの向上が期待できる一方で、著作権やプライバシー保護に関する懸念も存在します。これらの課題を克服するための適切な政策や対策を提案します。

    最後に、大学の卒業生や研究者、学術機関など関係者がオープンアクセスシステムを活用して学術情報を効果的に共有し、情報構築に貢献するための指針やベストプラクティスをまとめます。また、オープンアクセスの普及に向けた啓蒙活動や社会的な意識の醸成の重要性にも言及します。

    本論文の研究結果は、大学や学術機関の意思決定者や学術情報の利用者にとって貴重な情報となり、オープンアクセスシステムを通じた情報共有の重要性を理解し、学術コミュニケーションの発展に寄与することが期待されます。
    """
    
    title, content = randomChoice()
    
    print(title)
    keywords = tfidf([title])
    print(keywords)
    
    keywords = tfidf([content])
    print(keywords)
        
if __name__ == "__main__":
    main()