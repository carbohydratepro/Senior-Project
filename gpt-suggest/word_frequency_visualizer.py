import tkinter as tk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# NLTKのストップワードリストをダウンロード
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# GUIの作成
root = tk.Tk()
canvas = tk.Canvas(root)
canvas.pack()

# 文書の定義（ここでは例として固定の文書を使用）
text = """
Your text goes here. It can be any length and can include
any number of different words. The program will analyze
the frequency of each word and display the most common
words in a GUI window. The size of the word will correspond
to its frequency in the text.
"""

# ストップワード（a, theなど意味の少ない単語）の除去
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(text)
filtered_text = [w for w in word_tokens if not w in stop_words]

# 単語の出現回数をカウント
word_counts = Counter(filtered_text)

# 単語を出現回数順に並べ替え
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# GUIに単語を表示（出現回数が多い単語ほど大きく）
for word, count in sorted_words:
    size = count * 5  # フォントサイズを調整（適宜変更可能）
    label = tk.Label(root, text=word, font=("Helvetica", size))
    label.pack()

# GUIの表示
root.mainloop()
