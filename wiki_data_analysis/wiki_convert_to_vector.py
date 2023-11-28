import os
import re
import time
from tqdm import tqdm

def extract_info(text):
    # <doc>タグ内のid, title, 本文を抽出する正規表現
    pattern = r'<doc id="(.*?)" url=".*?" title="(.*?)">(.*?)</doc>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

wiki_directory = "./wiki"

count = 0

for root, dirs, files in os.walk(wiki_directory):
    for file in tqdm(files):
        # 拡張子がないファイルを対象とする
        if '.' not in file:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                extracted_info = extract_info(text)
                for id, title, content in extracted_info:
                    count += 1
                    
print(count)
