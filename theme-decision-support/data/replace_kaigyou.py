def remove_newlines(filename):
    # ファイルを読み込む
    with open(filename, 'r', encoding="utf-8") as file:
        text = file.read()

    # 改行を削除する
    text = text.replace('\n', '')

    # ファイルに書き戻す
    with open(filename, 'w', encoding="utf-8") as file:
        file.write(text)

# 使用例
remove_newlines('./theme-decision-support/data/replace.txt')
