from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from janome.tokenizer import Tokenizer
from database import Db
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import random

# ログの設定
logging.basicConfig(level=logging.INFO)
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['font.size'] = 5  # フォントサイズを10に設定


def get_data(dbname):
    command = 'SELECT * from theses'
    db = Db(dbname)
    data = db.db_output(command)
    return data

# データセットからランダムに抽出する関数
def select_random_elements(array, n):
    if n > len(array):
        raise ValueError("n is greater than the length of the array.")
    
    random_elements = random.sample(array, n)
    return random_elements
    
def tsne_plt(documents):
    t = Tokenizer()

    # TaggedDocumentの形にする
    logging.info("create tagged doccument...")
    tagged_documents = [TaggedDocument(words=[token.surface for token in t.tokenize(d[1])], tags=[d[0]]) for d in documents]

    # Train the Doc2Vec model
    logging.info("Train the Doc2Vec model...")
    model = Doc2Vec(tagged_documents, dm=0, vector_size=300, window=15, alpha=.025,min_alpha=.025, min_count=1, sample=1e-6, workers=4, epochs=100)

    # Get the document vectors from the model
    logging.info("Get the document vectors from the model")
    X = model.dv.vectors

    # Apply t-SNE to the document vectors
    logging.info("Apply t-SNE to the document vectors")
    tsne = TSNE(n_components=2, random_state=0, perplexity=100)  # Adjust perplexity value here

    X_tsne = tsne.fit_transform(X)

    # Get the title of documents from the 'documents' variable
    titles = [doc.tags[0] for doc in tagged_documents]

    # Create a scatter plot of the t-SNE output
    plt.figure(figsize=(16, 16))

    for i in range(len(X_tsne)):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1])
        plt.annotate(titles[i], xy=(X_tsne[i, 0], X_tsne[i, 1]), xytext=(5, 2), 
                    textcoords='offset points', ha='right', va='bottom')
    plt.show()
    
    model.save(f'./gpt-suggest/model/my_model')
    

def get_similar(documents):
    model = Doc2Vec.load(f'./gpt-suggest/model/my_model')
    
    # 文書IDのリストを作成
    doc_ids = list(range(len(documents)))

    # ランダムに文書IDを選択
    random_id = random.choice(doc_ids)

    # 選択した文書のタイトルと内容を出力
    print(f"選択した論文のタイトル: {documents[random_id][0]}")

    # 選択した文書と最も類似した5つの文書を取得
    similar_docs = model.dv.most_similar([random_id], topn=5)

    print("\n類似した論文のタイトル:")
    for doc_title, similarity in similar_docs:
        print(f"{doc_title} (類似度: {similarity})")


def main():
    documents = []

    dbname = './gpt-suggest/db/tuboroxn.db'
    dataset = get_data(dbname)
    for data in tqdm(dataset):
        if len(data[-1]) > 200 and data[-2] != r"卒業論文\n\n論文題目\n\n([\s\S]*?)\n\n":
            documents.append([data[-2], data[-1]]) # .replace("\n", "").replace("・", "")
        else:
            pass

    
    documents = select_random_elements(documents, 200)
    # model = tsne_plt(documents)
    
    get_similar(documents)
    
if __name__ == "__main__":
    main()