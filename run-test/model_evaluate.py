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
    def __init__(self, vocab_size=30522, hidden_size=768):
        super(Seq2SeqModel, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.LSTM(768, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.start_token = torch.zeros((1, 1, hidden_size))  # start_token初期化

        # エンコーダの出力をデコーダの入力に変換する全結合層を追加
        self.encoder_to_decoder = nn.Linear(768, hidden_size)

    def forward(self, input_ids, decoder_input=None, h=None, c=None):
        encoder_output = self.encoder(input_ids)[0]

        if decoder_input is None:
            decoder_input = encoder_output

        # エンコーダの出力をデコーダの入力に変換
        decoder_input = self.encoder_to_decoder(decoder_input)
        
        # LSTMは(hidden_state, cell_state)のタプルを返す
        decoder_output, (h, c) = self.decoder(decoder_input, (h, c)) if h is not None and c is not None else self.decoder(decoder_input)
        output = self.fc(decoder_output)

        return output, (h, c)


    def generate(self, input_ids, decoder_input):
        encoder_output = self.encoder(input_ids)[0]
        decoder_output, _ = self.decoder(decoder_input)  # use decoder input
        output = self.fc(decoder_output)
        return output.argmax(2)


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
        problem_encoding = {k: v.to(device) for k, v in problem_encoding.items() if k != 'token_type_ids' and k != 'attention_mask'} # Exclude 'token_type_ids' and 'attention_mask'

        output_tokens = []
        h, c = None, None
        for _ in range(max_length):
            output, (h, c) = model(problem_encoding['input_ids'], None, h, c)
            output_token = output.argmax(2).unsqueeze(1)  # Change dimension from 1 to 2
            output_tokens.append(output_token)

        output_tokens = torch.cat(output_tokens, dim=1)
        program = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return program



def tokens_to_code(tokenizer, tokens):
    """
    BERT tokenizerでエンコードされたトークンIDのリストを、元のコードに戻す関数
    """
    return tokenizer.decode(tokens)

def eval2():
    vocab_size = 30522
    generated = []

    # データセットの読み込み
    data = read_data(100)
    datasets = create_datasets(data)
    test_problems = [data[0] for data in datasets]
    test_programs = [data[1] for data in datasets]

    # テストデータの準備
    padding_token_id = 0
    test_dataset = Seq2SeqDataset(test_problems, test_programs, padding_token_id)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

    # モデルの読み込み
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Seq2SeqModel().to(device)
    model.load_state_dict(torch.load('.\\run-test\\checkpoint\\model_ak.pth'))

    # モデルをデバイスに移動
    model = model.to(device)

    # BERT tokenizerのロード
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # モデルの評価
    model.eval()
    for problems, programs in test_dataloader:
        total_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            output = model(problems.to(device), programs.to(device))
            loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), programs.to(device).view(-1))
            total_loss += loss.item()

            # 予測を取得（最大値のインデックス）
            _, predicted = torch.max(output, dim=-1)

            # LSTMの出力を配列に格納
            generated.append([predicted.tolist(), tokens_to_code(tokenizer, predicted.tolist())])

            # 精度を計算
            correct = (predicted == programs.to(device)).float()
            accuracy = correct.sum() / len(correct)
            total_accuracy += accuracy.item()

    # 平均損失と精度を出力
    print('Test Loss:', total_loss / len(test_dataloader))
    print('Test Accuracy:', total_accuracy / len(test_dataloader))

    return generated

if __name__ == "__main__":
    generated_program = eval2()
    while True:
        user_input = input("input:")
        if user_input == "q":
            break
        else:
            index_num = int(user_input)
            print("LSTM output (token IDs):", generated_program[index_num][0])
            print("LSTM output (code):", generated_program[index_num][1])
