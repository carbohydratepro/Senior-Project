def remove_newlines(filename):
    # ファイルを読み込む
    with open(filename, 'r', encoding="utf-8") as file:
        text = file.read()

    # 改行を削除する
    text = text.replace('\n', '')

    # ファイルに書き戻す
    with open(filename, 'w', encoding="utf-8") as file:
        file.write(text)
    
    return text

def append_to_csv(filename, text):
    with open(filename, 'a', encoding="utf-8") as file:
        file.write(text + '\n')



# 使用例
text = remove_newlines('./theme-decision-support/data/replace.txt')


# 使用例
# append_to_csv('./theme-decision-support/data/contents.csv', text)
