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

def read_data(data_num=100): 
    # データセットからランダムに抽出する関数
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

    data = cur.fetchall() #datas=[[id, problem, program], [id, problem, program], [id, problem, program], ...]

    # 接続を閉じる
    conn.close()

    return select_random_elements(data, data_num)


def create_datasets(data):
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

    # textの言語を判定
    def detect_language(text):
        try:
            lang = detect(text)
            return lang
        except:
            return "Could not detect language"



    datasets = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, data in enumerate(tqdm(data, postfix="データセットをロード中")):
        if detect_language(data[1]) == "en":
            problem_encoding = tokenizer.encode_plus(
                data[1],
                truncation=True,
                padding='max_length',
                max_length=512
                )
                    
            ast_tree = code_to_ast(data[2])
            if ast_tree != None:
                feature_vector = ast_to_feature_vector(ast_tree, max_length=512)
                program_encoding = tokenizer.encode_plus(
                    feature_vector,
                    truncation=True,
                    padding='max_length',
                    max_length=512
                    )
                datasets.append([problem_encoding, program_encoding])
            else:
                continue
    
    return datasets

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
    loss_values = []  # 追加：損失値を保存するリスト
    accuracy_values = []  # 追加：精度を保存するリスト

    data = read_data(100)
    datasets = create_datasets(data) # datasets：[[problem1, program1-1], [problem1, program1-2], ...[problem1, program1-n], [problem2, program2-1], ...]
    problems = [data[0] for data in datasets]
    programs = [data[1] for data in datasets]

    # データローダの準備
    dataset = Seq2SeqDataset(problems, programs, padding_token_id) # problemsとprogramsはそれぞれ問題文と正答プログラムのリスト
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    # モデルとオプティマイザの準備
    model = Seq2SeqModel().to(device)  # deviceはハードウェア（CPUまたはGPU）
    optimizer = Adam(model.parameters(), lr=0.00001)


    # モデルの訓練
    for epoch in range(num_epochs):  # num_epochsはエポック数
        total_predictions = 0
        total_correct_predictions = 0
        for problems, programs in dataloader:
            optimizer.zero_grad()
            output = model(problems.to(device), programs.to(device))  # Move tensors to GPU
            loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), programs.to(device).view(-1))  # Move tensors to GPU
            loss.backward()
            optimizer.step()

            # バッチ内の全てのサンプルに対して予測を計算
            _, predicted = output.view(-1, vocab_size).max(dim=1)
            # バッチ内の全てのサンプルに対して正解ラベルを取得
            true = programs.view(-1).to(device)
            # パディングトークンを無視して、バッチ内で予測が正解した数を計算
            non_padding_mask = true.ne(padding_token_id)
            correct_predictions = (predicted[non_padding_mask] == true[non_padding_mask]).sum().item()

            total_predictions += non_padding_mask.sum().item()
            total_correct_predictions += correct_predictions

        accuracy = total_correct_predictions / total_predictions
        print('Epoch:', epoch, 'Loss:', loss.item(), 'Accuracy:', accuracy)

        loss_values.append(loss.item())
        accuracy_values.append(accuracy)


    torch.save(model.state_dict(), f'.\\run-test\\checkpoint\\model_ak.pth')
    eval()

def eval():
    vocab_size = 30522

    # データセットの読み込み
    data = read_data(100)
    datasets = create_datasets(data) # datasets：[[problem1, program1-1], [problem1, program1-2], ...[problem1, program1-n], [problem2, program2-1], ...]
    test_problems = [data[0] for data in datasets]
    test_programs = [data[1] for data in datasets]

    # テストデータの準備
    padding_token_id = 0
    test_dataset = Seq2SeqDataset(test_problems, test_programs, padding_token_id)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

    # モデルの読み込み
    # Load
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルの準備
    model = Seq2SeqModel().to(device)  # deviceはハードウェア（CPUまたはGPU）

    # Move model to the GPU if available
    model.load_state_dict(torch.load('.\\run-test\\checkpoint\\model_ak.pth'))
 
    # モデルをデバイスに移動
    model = model.to(device)

    # モデルの評価
    model.eval()  # モデルを評価モードに
    for problems, programs in test_dataloader:
        total_loss = 0
        total_accuracy = 0
        with torch.no_grad():  # 勾配の計算をオフ
            output = model(problems.to(device), programs.to(device))
            loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), programs.to(device).view(-1))
            total_loss += loss.item()

            # 予測を取得（最大値のインデックス）
            _, predicted = torch.max(output, dim=-1)

            # 精度を計算
            correct = (predicted == programs.to(device)).float()  # 正解は1、不正解は0
            accuracy = correct.sum() / len(correct)
            total_accuracy += accuracy.item()

    # 平均損失と精度を出力
    print('Test Loss:', total_loss / len(test_dataloader))
    print('Test Accuracy:', total_accuracy / len(test_dataloader))


if __name__ == "__main__":
    main()