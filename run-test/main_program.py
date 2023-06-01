import sqlite3
import ast
import torch
import random
import matplotlib.pyplot as plt
from langdetect import detect
from tqdm import tqdm
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

    # データセットからランダムに抽出する関数
    def select_random_elements(array, n):
        if n > len(array):
            raise ValueError("n is greater than the length of the array.")
        
        random_elements = random.sample(array, n)
        return random_elements

    def detect_language(text):
        try:
            lang = detect(text)
            return lang
        except:
            return "Could not detect language"


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
    data_num = 0
    loss_values = []  # 追加：損失値を保存するリスト
    accuracy_values = []  # 追加：精度を保存するリスト

    # データセットの読み込み
    datasets = create_datasets(data_num)
    problems = [data[0] for data in tqdm(datasets, postfix="データセット処理中：プロブレム")]
    programs = [data[1] for data in tqdm(datasets, postfix="データセット処理中：プログラム")]

    # データローダの準備
    dataset = Seq2SeqDataset(problems, programs, padding_token_id) # problemsとprogramsはそれぞれ問題文と正答プログラムのリスト
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    # モデルとオプティマイザの準備
    model = Seq2SeqModel().to(device)  # deviceはハードウェア（CPUまたはGPU）
    optimizer = Adam(model.parameters(), lr=0.00001) #0.001


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

    # for epoch in range(num_epochs):  # num_epochsはエポック数
    #     total_loss = 0  # 追加：エポックごとの損失合計
    #     correct_predictions = 0  # 追加：正確な予測の数
    #     total_predictions = 0  # 追加：予測の総数

    #     for problems, programs in dataloader:
    #         optimizer.zero_grad()
    #         output = model(problems.to(device), programs.to(device))  # Move tensors to GPU
    #         loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), programs.to(device).view(-1))  # Move tensors to GPU
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()
    #         correct_predictions += (output.argmax(1) == programs.to(device)).sum().item()
    #         total_predictions += programs.size(0)

    #     epoch_loss = total_loss / len(dataloader)  # エポックごとの平均損失
    #     epoch_accuracy = correct_predictions / total_predictions  # エポックごとの精度

    #     print('Epoch:', epoch, 'Loss:', epoch_loss, 'Accuracy:', epoch_accuracy)

    #     # 損失と精度を保存
    #     loss_values.append(epoch_loss)
    #     accuracy_values.append(epoch_accuracy)


    # 損失と精度のグラフを表示
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_values, label='Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_values, label='Accuracy')
    plt.legend()
    plt.show()

    # グラフを画像として保存
    plt.savefig(f".\\run-test\\figure\\training_graph10000ak.png")

    # Save
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'.\\run-test\\checkpoint\\checkpoint_10000ak.pth')
    
    torch.save(model.state_dict(), f'.\\run-test\\checkpoint\\model_10000ak.pth')


def evaluate(model, dataloader, device, criterion):
    vocab_size = 30522
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for problems, programs in dataloader:
            output = model(problems.to(device), programs.to(device))  # Move tensors to GPU
            loss = criterion(output.view(-1, vocab_size), programs.to(device).view(-1))  # Move tensors to GPU
            total_loss += loss.item()
    return total_loss / len(dataloader)

def generate(model, problem, tokenizer, device, max_length=512):
    model.eval()
    with torch.no_grad():
        problem_encoding = tokenizer.encode_plus(
            problem,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        problem_encoding = {k: v.to(device) for k, v in problem_encoding.items()}
        output = model.generate(**problem_encoding, max_length=max_length)
        program = tokenizer.decode(output[0], skip_special_tokens=True)
    return program


def eval():
    # データセットの読み込み
    datasets = create_datasets()
    test_problems = [data[0] for data in tqdm(datasets, postfix="データセット処理中：プロブレム")]
    test_programs = [data[1] for data in tqdm(datasets, postfix="データセット処理中：プログラム")]

    # テストデータの準備
    padding_token_id = 0
    test_dataset = Seq2SeqDataset(test_problems, test_programs, padding_token_id)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

    # モデルの読み込み
    # Load
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # モデルとオプティマイザの準備
    model = Seq2SeqModel().to(device)  # deviceはハードウェア（CPUまたはGPU）
    optimizer = Adam(model.parameters(), lr=0.001)

    # Load on CPU
    checkpoint = torch.load(f'.\\run-test\\checkpoint\\checkpoint_100.pth', map_location=torch.device('cuda'))

    # Move model to the GPU if available
    model.load_state_dict(checkpoint['model_state_dict'])

    # Similarly, ensure that the optimizer state is on the right device
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # モデルの評価
    model.eval()  # モデルを評価モードに
    total_loss = 0
    total_accuracy = 0
    vocab_size=30522
    for problems, programs in test_dataloader:
        with torch.no_grad():  # 勾配の計算をオフ
            output = model(problems.to(device), programs.to(device))
            loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), programs.view(-1))
            total_loss += loss.item()

            # 予測を取得（最大値のインデックス）
            _, predicted = torch.max(output, dim=-1)

            # 精度を計算
            correct = (predicted == programs).float()  # 正解は1、不正解は0
            accuracy = correct.sum() / len(correct)
            total_accuracy += accuracy.item()

    # 平均損失と精度を出力
    print('Test Loss:', total_loss / len(test_dataloader))
    print('Test Accuracy:', total_accuracy / len(test_dataloader))


if __name__ == "__main__":
    main()
