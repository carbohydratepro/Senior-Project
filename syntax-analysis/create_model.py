from database import Db
from create_data import isFile
from janome.tokenizer import Tokenizer
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec


##モデルの
class Model():
    def __init__(self, modelname):
        self.modelname=modelname

    # 保存
    def save(self, model):
        model.save(self.modelname)

    # 読み込み
    def read(self):
        return Doc2Vec.load(self.modelname)

    # 削除
    def delete(self):
        pass



def create(modelname):
    dbname = './syntax-analysis/db/coding_problems.db'
    db = Db(dbname)
    documents = [(data[2], data[1]) for data in db.db_output()] #二次元配列で文章を格納

    t = Tokenizer()

    created_data = []

    for strings in tqdm(documents):
        created_data.append(TaggedDocument([token.surface for token in t.tokenize(strings[0])], [strings[1]]))

    model = Doc2Vec(created_data,  dm=0, vector_size=300, window=15, alpha=.025,min_alpha=.025, min_count=1, sample=1e-6)
    Model(modelname).save(model)


def ratingAverage(num): #num：配列
    return sum(num)/len(num)

def main():
    modelname = './syntax-analysis/model/coding_problems.model'

    if not isFile(modelname):
        create(modelname)
    
    


if __name__ == "__main__":
    main()