import sqlite3
import ast
from langdetect import detect
from tqdm import tqdm
import torch
import random
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

def create_datasets(data_num=100): 
    # コードをASTに変換
    def code_to_ast(code):
        try:
            tree = ast.parse(code)
            return tree
        except Exception as e:
            print("Error in parsing:", e)
            return None

    # ASTから特徴ベクトルへ変換
    def ast_to_feature_vector(ast_tree, max_length):
        # DFSを使用してASTツリーを訪問し、ノードタイプ名を収集
        nodes = list(ast.walk(ast_tree))
        feature_vector = [type(n).__name__ for n in nodes]

        # 特徴ベクトルの長さがmax_lengthを超えている場合は切り捨て
        if len(feature_vector) > max_length:
            feature_vector = feature_vector[:max_length]
        # 特徴ベクトルの長さがmax_lengthに満たない場合はパディング
        else:
            feature_vector += ['<PAD>'] * (max_length - len(feature_vector))

        return feature_vector



    def detect_language(text):
        try:
            lang = detect(text)
            return lang
        except:
            return "Could not detect language"

    def select_random_elements(array, n):
        if n > len(array):
            raise ValueError("n is greater than the length of the array.")
        
        random_elements = random.sample(array, n)
        return random_elements

    # データベースに接続
    conn = sqlite3.connect(".\syntax-analysis\db\mydatasets.db")

    # カーソルオブジェクトを作成
    cur = conn.cursor()

    # SQLクエリを実行
    print("データベースからデータを読み込み中")
    cur.execute('''
        SELECT problems.problem_id, problems.problem, programs.program
        FROM problems
        INNER JOIN programs ON problems.problem_id = programs.problem_id
    ''')

    datas = cur.fetchall() #datas=[[id, problem, program], [id, problem, program], [id, problem, program], ...]
    datasets = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, data in enumerate(tqdm(datas, postfix="データセットをデータベースからロード中")):
        if detect_language(data[1]) == "en":
            problem_encoding = tokenizer.encode_plus(
                data[1],
                truncation=True,
                padding='max_length',
                max_length=512
                )
                    
            
            program_encoding = tokenizer.encode_plus(
                data[2],
                truncation=True,
                padding='max_length',
                max_length=512
                )
            datasets.append([problem_encoding, program_encoding])
        


    # 接続を閉じる
    conn.close()
    return select_random_elements(datasets, data_num)



class Seq2SeqDataset(Dataset):
    # データセットを管理するためのクラス
    def __init__(self, problems, programs, padding_token_id):
        self.problems = problems
        self.programs = programs
        self.padding_token_id = padding_token_id

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx], self.programs[idx]
    
    def collate_fn(self, batch):
        problems, programs = zip(*batch)
        problems = [torch.tensor(p['input_ids']) for p in problems]
        programs = [torch.tensor(p['input_ids']) for p in programs]
        problems = pad_sequence(problems, batch_first=True, padding_value=self.padding_token_id)
        programs = pad_sequence(programs, batch_first=True, padding_value=self.padding_token_id)
        return problems, programs



# モデルの定義
class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.LSTM(768, 768, batch_first=True)
        self.fc = nn.Linear(768, 30522)  # vocab_size=30522：ボキャブラリサイズ（トークンの数）

    def forward(self, input_ids, decoder_input):
        encoder_output = self.encoder(input_ids)[0]
        decoder_output, _ = self.decoder(encoder_output, None)  # use encoder output
        output = self.fc(decoder_output)
        return output


def main():
    # 初期情報の設定
    device = "cuda"
    num_epochs = 10
    vocab_size = 30522
    padding_token_id = 0

    # データセットの読み込み
    datasets = create_datasets(300)
    problems = [data[0] for data in tqdm(datasets, postfix="データセット処理中：プロブレム")]
    programs = [data[1] for data in tqdm(datasets, postfix="データセット処理中：プログラム")]

    # データローダの準備
    dataset = Seq2SeqDataset(problems, programs, padding_token_id) # problemsとprogramsはそれぞれ問題文と正答プログラムのリスト
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    # モデルとオプティマイザの準備
    model = Seq2SeqModel().to(device)  # deviceはハードウェア（CPUまたはGPU）
    optimizer = Adam(model.parameters(), lr=0.001)

    # モデルの訓練
    for epoch in range(num_epochs):  # num_epochsはエポック数
        for problems, programs in dataloader:
            optimizer.zero_grad()
            output = model(problems.to(device), programs.to(device))  # Move tensors to GPU
            loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), programs.to(device).view(-1))  # Move tensors to GPU
            loss.backward()
            optimizer.step()


        print('Epoch:', epoch, 'Loss:', loss.item())


if __name__ == "__main__":
    main()
