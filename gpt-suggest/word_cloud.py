from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from database import Db
import matplotlib.pyplot as plt


def get_data(dbname):
    command = 'SELECT * from theses'
    db = Db(dbname)
    data = db.db_output(command)
    return data

def create_wordcloud(text):
    # ストップワードの設定
    stopwords = ["の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと", "として", "い", "や", "れる", "など", "なっ", "ない", "この", "ため", "その", "あっ", "よう", "また", "もの", "という", "あり", "まで", "られ", "なる", "へ", "か", "だ", "これ", "によって", "により", "おり", "より", "による", "ず", "なり", "られる", "において", "ば", "なかっ", "なく", "しかし", "について", "せ", "だっ", "その後", "できる", "それ", "う", "ので", "なお", "のみ", "でき", "き", "つ", "における", "および", "いう", "さらに", "でも", "ら", "たり", "その他", "に関する", "たち", "ます", "ん", "なら", "に対して", "特に", "せる", "及び", "これら", "とき", "では", "にて", "ほか", "ながら", "うち", "そして", "とともに", "ただし", "かつて", "それぞれ", "または", "に対する", "ほとんど", "と共に", "といった", "です", "とも", "ところ", "ここ"]

    # Janomeトークナイザのインスタンス化
    t = Tokenizer()

    # フィルターの設定
    char_filters = [UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(u'蛇の目', u'janome')]
    token_filters = [CompoundNounFilter(), POSStopFilter(['名詞,非自立', '助詞', '助動詞', '動詞,非自立']), LowerCaseFilter(), ExtractAttributeFilter('base_form')]
    a = Analyzer(char_filters=char_filters, tokenizer=t, token_filters=token_filters)

    # 単語を分割
    words = list(a.analyze(text))

    # 単語をスペースでつなげる
    word_chain = " ".join(words)

    # ワードクラウドの生成
    wordcloud = WordCloud(
        font_path="C:\\Windows\\Fonts\\meiryo.ttc",
        background_color="white",
        stopwords=set(stopwords),
        width=800,
        height=600
        ).generate(word_chain)
    wordcloud.to_file("./gpt-suggest/picture/wordcloud.png")

    # ワードクラウドの表示
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def main():
    
    dbname = './gpt-suggest/db/tuboroxn.db'
    data = get_data(dbname)
    # 使用例
    text = data[2][-1]
    create_wordcloud(text)

if __name__ == "__main__":
    main()

# アフリカの水辺にはクロコダイルがいます。
# 国や水辺、クロコダイルは選択肢として用意しておく。