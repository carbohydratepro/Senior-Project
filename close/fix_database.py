import sqlite3

# データベースに接続
conn = sqlite3.connect('E:/words_embeddings_from_mask.db')
cursor = conn.cursor()

try:
    # 条件に一致するデータを検索し、削除するSQLクエリ
    query = """
    DELETE FROM word_embeddings
    WHERE content = CHAR(10) || word || CHAR(10) || CHAR(10) || CHAR(10);
    """
    cursor.execute(query)
    conn.commit()
    print("該当するデータを削除しました。")
except Exception as e:
    print("エラーが発生しました：", e)
finally:
    conn.close()
