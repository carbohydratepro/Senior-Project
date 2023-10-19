import requests
from bs4 import BeautifulSoup

def get_weblio_definition(word):
    url = f"https://www.weblio.jp/content/{word}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page for {word}. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Weblioのページ構造に基づいて意味を抽出
    definition_section = soup.find('td', class_='content-explanation')

    if not definition_section:
        print(f"Definition not found for {word}")
        return None
    
    return definition_section.get_text()

word = "例"
definition = get_weblio_definition(word)
print(definition)
