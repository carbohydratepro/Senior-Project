import sqlite3
from collections import Counter, defaultdict
from janome.tokenizer import Tokenizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import numpy as np
from tqdm import tqdm


percentiles = [25, 50, 75, 90]

def calculate_thresholds(word_counts, percentiles):
    """
    指定されたパーセンタイルに基づいて閾値を計算する。

    Args:
    word_counts (list of int): 単語の出現回数のリスト
    percentiles (list of int): 使用するパーセンタイル

    Returns:
    list of int: 計算された閾値
    """
    return [np.percentile(word_counts, percentile) for percentile in percentiles]

def assign_weight_category(weight, thresholds):
    """
    指定された重みに基づいてカテゴリ（1から5）を割り当てる。

    Args:
    weight (int): 単語の出現回数
    thresholds (list of int): 閾値のリスト

    Returns:
    int: 単語の重みカテゴリ
    """
    for i, threshold in enumerate(thresholds):
        if weight <= threshold:
            return i + 1
    return len(thresholds) + 1

def count_words_in_db(db_path, num_words):
    """
    指定されたSQLiteデータベース内のword_countsテーブルから単語とその出現回数を取得する。

    Args:
    db_path (str): データベースファイルのパス

    Returns:
    list of dict: 各単語とその出現回数を含む辞書のリスト
    """
    # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # word_countsテーブルから単語と出現回数を取得
    cursor.execute("SELECT word, count FROM word_counts")
    word_counts = cursor.fetchall()

    # 単語の出現回数が5以上の単語のみをフィルタリング
    word_counts = [(word, count) for word, count in word_counts if count > 5]

    # 指定された数の単語をランダムに選択
    random_words = random.sample(word_counts, min(num_words, len(word_counts)))

    # 単語の出現回数のリストを取得
    word_counts = [count for _, count in random_words]

    # 閾値を計算
    thresholds = calculate_thresholds(word_counts, percentiles)

    # 単語とその出現回数を辞書のリストとして作成し、重みを5段階評価に変換
    words_weighted = [{"text": word, "weight": assign_weight_category(count, thresholds)} for word, count in random_words]
    
    return words_weighted


# 使用例
# db_path = 'path_to_your_database.db'
# words = count_words_in_db(db_path)
# print(words)

def store_word_counts(db_path):
    """
    Janomeを使用して、指定されたSQLiteデータベース内のthesesテーブルのcontentカラムからテキストを取得し、
    形態素解析を行い、単語の出現回数をカウントして新しいテーブルに格納する。

    Args:
    db_path (str): データベースファイルのパス
    """
    stop_words = [
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も",
        "する", "から", "な", "こと", "として", "い", "や", "れる", "など", "なっ", "ない", "この", "ため",
        "その", "あっ", "よう", "また", "もの", "という", "あり", "まで", "られ", "なる", "へ", "か", "だ",
        "これ", "によって", "により", "おり", "より", "による", "ず", "なり", "られる", "において", "ば",
        "なかっ", "なく", "しかし", "について", "せ", "だっ", "その後", "できる", "それ", "う", "ので",
        "なお", "のみ", "でき", "き", "つ", "における", "および", "いう", "さらに", "でも", "ら", "たり",
        "その他", "に関する", "たち", "ます", "ん", "なら", "に対して", "特に", "せる", "及び", "これら",
        "とき", "では", "にて", "ほか", "ながら", "うち", "そして", "とともに", "ただし", "かつて", "それぞれ",
        "または", "に対する", "ほとんど", "と共に", "といった", "です", "とも", "ところ", "ここ",
        ".", ":", ", ", "．", "，", "。", "、", "(", ")", "（", "）", "cid", "/", "[", "]",
        ]

    # 全角および半角の記号を含む正規表現パターン
    symbol_pattern = re.compile(r"[！”＃＄％＆’（）＝～｜‘｛＋＊｝＜＞？＿!#$%&'()=~|'{+*}<>?_1234567890!-\/:-@\[-`{-~、。〃〄々〆〇〈〉《》「」『』【】〒〓〔〕〖〗〘〙〚〛〜〝〞〟〠〡〢〣〤〥〦〧〨〩〪〭〮〯〫〬〰〱〲〳〴〵〶〷〸〹〺〻〼〽〾〿1234567890１２３４５６７８９０]")
    
    # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 新しいテーブルを作成（すでに存在する場合は無視）
    cursor.execute("CREATE TABLE IF NOT EXISTS word_counts (word TEXT, count INTEGER)")

    # thesesテーブルからcontentカラムのデータを取得
    cursor.execute("SELECT content FROM theses")
    contents = cursor.fetchall()

    # Janomeのトークナイザーを初期化
    tokenizer = Tokenizer()

    # 単語の出現回数をカウント
    words_count = Counter()
    for content in contents:
        if content[0]:
            tokens = tokenizer.tokenize(content[0])
            words_count.update(
                token.surface for token in tokens 
                if token.part_of_speech.startswith('名詞') 
                and token.surface not in stop_words
                and not symbol_pattern.search(token.surface)
            )

    # 既存の単語カウントを削除
    cursor.execute("DELETE FROM word_counts")

    # 単語とその出現回数を新しいテーブルに挿入
    for word, count in words_count.items():
        cursor.execute("INSERT INTO word_counts (word, count) VALUES (?, ?)", (word, count))

    # 変更をコミット
    conn.commit()
    conn.close()




def store_word_tfidf(db_path):
    """
    tfidfを使用して、指定されたSQLiteデータベース内のthesesテーブルのcontentカラムからテキストを取得し、
    重要単語を抽出し、新しいテーブルに格納する。

    Args:
    db_path (str): データベースファイルのパス
    """
            
    def tokenize(text):
        """
        日本語のテキストをトークナイズする関数。
        """
        tokenizer = Tokenizer()
        return [token.surface for token in tokenizer.tokenize(text)]
    
       # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 新しいテーブルを作成（すでに存在する場合は無視）
    cursor.execute("CREATE TABLE IF NOT EXISTS word_tfidf (word TEXT, total_evaluation REAL, count INTEGER)")

    # thesesテーブルからcontentカラムのデータを取得
    cursor.execute("SELECT content FROM theses")
    contents = [item[0] if item[0] is not None else "" for item in cursor.fetchall()]
        # TfidfVectorizerをカスタムトークナイザーで初期化
    vectorizer = TfidfVectorizer(tokenizer=tokenize)

    # 文書リストからTF-IDF行列を計算
    tfidf_matrix = vectorizer.fit_transform(contents)
    words = vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.sum(axis=0).A1

    # 単語の出現回数をカウント
    word_counts = Counter()
    for doc in contents:
        word_counts.update(tokenize(doc))
        

    cursor.execute("DELETE FROM word_tfidf")
    
    # 単語とその総評価値、カウントをデータベースに挿入
    for word in tqdm(words):
        total_evaluation = tfidf_values[words.tolist().index(word)]
        count = word_counts[word]
        cursor.execute("INSERT INTO word_tfidf (word, total_evaluation, count) VALUES (?, ?, ?)", (word, total_evaluation, count))

    # コミットとクローズ
    conn.commit()
    conn.close()
    


def top_tfidf_words(db_path):
    """
    データベースから単語のTF-IDFと出現回数を取得し、total_evaluation / count の値で上位20の単語を出力する。
    """
    # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # データを取得
    cursor.execute("SELECT word, total_evaluation, count FROM word_tfidf")
    words_data = cursor.fetchall()

    # total_evaluation / count を計算
    word_scores = [(word, total_evaluation / count) for word, total_evaluation, count in words_data if count > 0]

    # スコアでソートし、上位20の単語を取得
    top_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:100]

    # 結果を出力
    for word, score in top_words:
        print(f"{word}: {score}")

    # データベースを閉じる
    conn.close()


def tokenize(text, pattern, stop_words):
    """
    日本語のテキストをトークナイズし、指定された正規表現パターンとストップワードを考慮する関数。
    """
    tokenizer = Tokenizer()
    tokens = []
    for token in tokenizer.tokenize(text):
        token_text = token.surface
        if not re.match(pattern, token_text) and token_text not in stop_words:
            tokens.append(token_text)
    return tokens

def store_top_words_per_document(db_path):
    """
    各文書からTF-IDFの上位50単語を抽出し、データベースに保存する関数。
    ストップワードと正規表現パターンを考慮してトークナイズする。
    """
    
    stop_words = [
            "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も",
            "する", "から", "な", "こと", "として", "い", "や", "れる", "など", "なっ", "ない", "この", "ため",
            "その", "あっ", "よう", "また", "もの", "という", "あり", "まで", "られ", "なる", "へ", "か", "だ",
            "これ", "によって", "により", "おり", "より", "による", "ず", "なり", "られる", "において", "ば",
            "なかっ", "なく", "しかし", "について", "せ", "だっ", "その後", "できる", "それ", "う", "ので",
            "なお", "のみ", "でき", "き", "つ", "における", "および", "いう", "さらに", "でも", "ら", "たり",
            "その他", "に関する", "たち", "ます", "ん", "なら", "に対して", "特に", "せる", "及び", "これら",
            "とき", "では", "にて", "ほか", "ながら", "うち", "そして", "とともに", "ただし", "かつて", "それぞれ",
            "または", "に対する", "ほとんど", "と共に", "といった", "です", "とも", "ところ", "ここ",
            ".", ":", ", ", "．", "，", "。", "、", "(", ")", "（", "）", "cid", "/", "[", "]",
            ]

    # 全角および半角の記号を含む正規表現パターン
    symbol_pattern = re.compile(r"[！”＃＄％＆’（）＝～｜‘｛＋＊｝＜＞？＿!#$%&'()=~|'{+*}<>?_1234567890!-\/:-@\[-`{-~、。〃〄々〆〇〈〉《》「」『』【】〒〓〔〕〖〗〘〙〚〛〜〝〞〟〠〡〢〣〤〥〦〧〨〩〪〭〮〯〫〬〰〱〲〳〴〵〶〷〸〹〺〻〼〽〾〿1234567890１２３４５６７８９０]")
    
    # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブルを作成
    cursor.execute("CREATE TABLE IF NOT EXISTS document_top_words (word TEXT, tfidf REAL)")

    # thesesテーブルからcontentカラムのデータを取得
    cursor.execute("SELECT content FROM theses")
    contents = [item[0] if item[0] is not None else "" for item in cursor.fetchall()]
    
    cursor.execute("DELETE FROM document_top_words")
    
    # 各文書に対してTF-IDFを計算し、上位50単語を抽出
    for document in contents:
        vectorizer = TfidfVectorizer(tokenizer=lambda text: tokenize(text, symbol_pattern, stop_words))
        
        try:
            tfidf_matrix = vectorizer.fit_transform([document])
        except ValueError:
            # 空の語彙の場合、次の文書に進む
            continue

        feature_names = vectorizer.get_feature_names_out()
        sorted_indices = tfidf_matrix.toarray()[0].argsort()[::-1][:50]

        # 上位50単語をデータベースに挿入
        for idx in sorted_indices:
            word = feature_names[idx]
            tfidf_value = tfidf_matrix[0, idx]
            cursor.execute("INSERT INTO document_top_words (word, tfidf) VALUES (?, ?)", (word, tfidf_value))

    # コミットとクローズ
    conn.commit()
    conn.close()
    
def print_top_tfidf_words(db_path, top_n=10):
    """
    データベースからTF-IDF値が最も高い上位の単語を表示する関数。
    """
    # データベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 上位の単語を取得
    cursor.execute("SELECT word, tfidf FROM document_top_words ORDER BY tfidf DESC LIMIT ?", (top_n,))
    top_words = cursor.fetchall()

    # 結果を出力
    for word, tfidf in top_words:
        print(f"{word}: {tfidf}")

    # データベースを閉じる
    conn.close()
    
if __name__ == "__main__":
    db_path = "./close/db/tuboroxn.db"
    print_top_tfidf_words(db_path, 100)
# 使用例
# db_path = 'path_to_your_database.db'
# store_word_counts(db_path)
