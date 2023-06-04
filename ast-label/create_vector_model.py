from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from datasets import read_data, create_datasets

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
    created_data = []

    for dataset in tqdm(datasets):
        created_data.append(TaggedDocument([token, label]))

    model = Doc2Vec(created_data,  dm=0, vector_size=300, window=15, alpha=.025,min_alpha=.025, min_count=1, sample=1e-6)
    Model(modelname).save(model)


def ratingAverage(num): #num：配列
    return sum(num)/len(num)

def main():
    modelname = './ast-label/model/PE0101.model'

    if not isFile(modelname):
        create(modelname)

    # model = Model(modelname).read()
    # sim = model.dv.most_similar('太宰治')
    # print(sim)
    # print(ratingAverage([s[1] for s in sim]))
def main():
    # 初期情報の設定
    data_num = 100

    # データセットの読み込み
    datasets = read_data(data_num)



if __name__ == "__main__":
    main()
    