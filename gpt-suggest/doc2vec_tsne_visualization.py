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

# # AIと量子コンピューティングの二つのトピック
# documents = [
#     ("Machine Learning in Automated Driving Systems", "AI", "この研究では、機械学習を用いた自動運転システムについて検討する。"),
#     ("Image Recognition with Deep Learning", "AI", "深層学習を用いた画像認識技術が近年注目されている。"),
#     ("The Role of AI in Future Industries", "AI", "AIは未来の産業をリードする重要な要素となる。"),
#     ("Natural Language Processing and Its Applications", "AI", "自然言語処理はAIの重要な研究分野であり、日常生活への応用が期待されている。"),
#     ("Cyber Security in The Age of AI", "AI", "AI時代のサイバーセキュリティについて考察する。"),
#     ("The Impact of AI on Employment", "AI", "AIが雇用に及ぼす影響についての調査研究"),
#     ("Understanding The Basics of Neural Networks", "AI", "ニューラルネットワークの基礎について学ぶ"),
#     ("The Use of AI in Healthcare", "AI", "医療現場でのAIの利用について"),
#     ("The Potential of Robotics in Manufacturing", "AI", "製造業におけるロボティクスの可能性"),
#     ("The Impact of Digital Transformation", "AI", "デジタル変革が社会に及ぼす影響"),
#     ("The Future of FinTech", "AI", "フィンテックの未来について"),
#     ("Advances in Facial Recognition Technology", "AI", "顔認識技術の進歩について"),
#     ("The Role of Smart Cities in Sustainable Development", "AI", "持続可能な開発におけるスマートシティの役割"),
#     ("Artificial Intelligence in Retail", "AI", "小売業界における人工知能の活用"),
#     ("The Use of Machine Learning in Agriculture", "AI", "農業における機械学習の利用"),
#     ("Predictive Analytics in Business", "AI", "ビジネスにおける予測分析の活用"),
#     ("AI Ethics and Governance", "AI", "AIの倫理とガバナンスについて"),
#     ("Robotics in Elderly Care", "AI", "高齢者ケアにおけるロボティクスの活用"),
#     ("The Application of AI in Cybersecurity", "AI", "サイバーセキュリティにおけるAIの応用"),
#     ("Challenges in Implementing AI Solutions", "AI", "AIソリューションを実装する際の課題"),
#     ("The Fundamentals of Quantum Computing", "Quantum Computing", "量子コンピューティングの基礎について"),
#     ("The Role of Quantum Computing in Future Industries", "Quantum Computing", "量子コンピューティングが未来の産業に果たす役割"),
#     ("Quantum Computing and Its Applications", "Quantum Computing", "量子コンピューティングとその応用"),
#     ("The Impact of Quantum Computing on Cybersecurity", "Quantum Computing", "量子コンピューティングがサイバーセキュリティに及ぼす影響"),
#     ("The Role of Quantum Computing in Artificial Intelligence", "Quantum Computing", "量子コンピューティングがAIに果たす役割"),
#     ("Challenges and Opportunities in Quantum Computing", "Quantum Computing", "量子コンピューティングにおける課題と機会"),
#     ("The Future of Quantum Computing", "Quantum Computing", "量子コンピューティングの未来について"),
#     ("Advances in Quantum Algorithms", "Quantum Computing", "量子アルゴリズムの進歩について"),
#     ("The Role of Quantum Mechanics in Computing", "Quantum Computing", "量子力学がコンピューティングに果たす役割"),
#     ("Quantum Computing for Solving Complex Problems", "Quantum Computing", "複雑な問題解決のための量子コンピューティング"),
# ]
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
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)  # Adjust perplexity value here

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
    plt.savefig(f"./gpt-suggest/picture/figure3.png")
    
    
def main():
    # documents = [
    #     ("Machine Learning in Automated Driving Systems", "この研究では、機械学習を用いた自動運転システムについて検討する。"),
    #     ("Image Recognition with Deep Learning", "深層学習を用いた画像認識技術が近年注目されている。"),
    #     ("The Role of AI in Future Industries", "AIは未来の産業をリードする重要な要素となる。"),
    #     ("Natural Language Processing and Its Applications", "自然言語処理はAIの重要な研究分野であり、日常生活への応用が期待されている。"),
    #     ("Cyber Security in The Age of AI", "AI時代のサイバーセキュリティについて考察する。"),
    #     ("The Impact of AI on Employment", "AIが雇用に及ぼす影響についての調査研究"),
    #     ("Understanding The Basics of Neural Networks", "ニューラルネットワークの基礎について学ぶ"),
    #     ("The Rise of Quantum Computing", "量子コンピューティングの台頭について"),
    #     ("The Influence of Blockchain Technology on Finance", "ブロックチェーン技術が金融業界に及ぼす影響"),
    #     ("Augmented Reality in Education", "教育現場における拡張現実(AR)技術の活用"),
    #     ("Virtual Reality in Gaming Industry", "ゲーム業界におけるバーチャルリアリティ(VR)の利用状況"),
    #     ("The Evolution of Cloud Computing", "クラウドコンピューティングの進化について"),
    #     ("Big Data and Its Impact on Business", "ビッグデータがビジネスに与える影響について"),
    #     ("The Emergence of 5G Technology", "5G技術の出現とその意味するもの"),
    #     ("The Role of Data Science in Decision Making", "意思決定におけるデータサイエンスの役割"),
    #     ("Understanding The Internet of Things (IoT)", "インターネット・オブ・シングス（IoT）について理解する"),
    #     ("The Use of AI in Healthcare", "医療現場でのAIの利用について"),
    #     ("The Potential of Robotics in Manufacturing", "製造業におけるロボティクスの可能性"),
    #     ("The Impact of Digital Transformation", "デジタル変革が社会に及ぼす影響"),
    #     ("The Future of FinTech", "フィンテックの未来について"),
    #     ("Advances in Facial Recognition Technology", "顔認識技術の進歩について"),
    #     ("The Role of Smart Cities in Sustainable Development", "持続可能な開発におけるスマートシティの役割"),
    #     ("Artificial Intelligence in Retail", "小売業界における人工知能の活用"),
    #     ("The Use of Machine Learning in Agriculture", "農業における機械学習の利用"),
    #     ("Predictive Analytics in Business", "ビジネスにおける予測分析の活用"),
    #     ("AI Ethics and Governance", "AIの倫理とガバナンスについて"),
    #     ("Robotics in Elderly Care", "高齢者ケアにおけるロボティクスの活用"),
    #     ("The Application of AI in Cybersecurity", "サイバーセキュリティにおけるAIの応用"),
    #     ("Challenges in Implementing AI Solutions", "AIソリューションを実装する際の課題"),
    #     ("The Future of Autonomous Vehicles", "自動運転車の未来について")
    # ]
    
    documents = []

    dbname = './gpt-suggest/db/tuboroxn.db'
    dataset = get_data(dbname)
    for data in tqdm(dataset):
        if len(data[-1]) > 200 and data[-2] != r"卒業論文\n\n論文題目\n\n([\s\S]*?)\n\n":
            documents.append([data[-2], data[-1]]) # .replace("\n", "").replace("・", "")
        else:
            pass

    
    documents = select_random_elements(documents, 50)
    tsne_plt(documents)
    
if __name__ == "__main__":
    main()