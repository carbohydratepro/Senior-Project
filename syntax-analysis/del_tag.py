import re

def delete_tag(text):
    tag = '<.+?>'  # タグを表す正規表現
    text = re.sub(tag, "", text)
    return text
    