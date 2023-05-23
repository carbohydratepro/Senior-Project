import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

class Seq2SeqDataset(Dataset):
    # データセットを管理するためのクラス
    def __init__(self, problems, programs):
        self.problems = problems
        self.programs = programs

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx], self.programs[idx]

# モデルの定義
class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.LSTM(768, 768, batch_first=True)
        self.fc = nn.Linear(768, vocab_size)  # vocab_size：ボキャブラリサイズ（トークンの数）

    def forward(self, input_ids, decoder_input):
        encoder_output = self.encoder(input_ids)[0]
        decoder_output, _ = self.decoder(decoder_input, encoder_output)
        output = self.fc(decoder_output)
        return output


# データローダの準備
dataset = Seq2SeqDataset(problems, programs)  # problemsとprogramsはそれぞれ問題文と正答プログラムのリスト
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデルとオプティマイザの準備
model = Seq2SeqModel().to(device)  # deviceはあなたのハードウェア（CPUまたはGPU）
optimizer = Adam(model.parameters(), lr=0.001)

# モデルの訓練
for epoch in range(num_epochs):  # num_epochsはエポック数
    for problems, programs in dataloader:
        optimizer.zero_grad()
        output = model(problems.to(device), programs.to(device))
        loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), programs.view(-1))
        loss.backward()
        optimizer.step()
    print('Epoch:', epoch, 'Loss:', loss.item())
