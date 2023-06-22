import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

def load_pretrained_word2vec(path):
    """事前学習済みのWord2Vecモデルをロードする"""
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model

def sentence_to_vector(sentence, model):
    """指定されたWord2Vecモデルを使用して、文をベクトルに変換する"""
    words = word_tokenize(sentence)
    word_vectors = [model[word] for word in words if word in model.key_to_index]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)


# 問題文を数値ベクトルに変換する関数（具体的な実装は使用する埋め込みに依存）
def text_to_vector(text):
    sentence_to_vector(text, w2v_model)

# ASTを特徴ベクトルに変換する関数
def ast_to_vector(ast):
    # DFSを使用してASTツリーを訪問し、ノードタイプ名を収集
    nodes = list(ast.walk(ast))
    feature_vector = [type(n).__name__ for n in nodes]
    return feature_vector

class ProblemDataset(Dataset):
    def __init__(self, problems, asts):
        self.problems = problems
        self.asts = asts

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return text_to_vector(self.problems[idx]), ast_to_vector(self.asts[idx])

class ProblemClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(ProblemClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def train(model, dataloader, criterion, optimizer):
    for epoch in range(10):
        for problems, asts in dataloader:
            outputs = model(problems)
            loss = criterion(outputs, asts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



# 事前学習済みのWord2Vecモデルのパス（ここではGoogleのモデルを例とします）
path_to_word2vec = '/path/to/GoogleNews-vectors-negative300.bin'

# Word2Vecモデルのロード
w2v_model = load_pretrained_word2vec(path_to_word2vec)

input_size = 300
output_size = 2
problems = [...]  # 問題文のリスト
asts = [...]  # ASTのリスト
dataset = ProblemDataset(problems, asts)
dataloader = DataLoader(dataset, batch_size=32)
model = ProblemClassifier(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

train(model, dataloader, criterion, optimizer)

