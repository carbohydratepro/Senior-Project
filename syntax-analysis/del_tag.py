import re

def delete_tag(text):
    if text is not None:
        tag = '<.+?>'  # タグを表す正規表現
        text = re.sub(tag, "", text)
        return text
    else:
        return None
    
# delete_tag()