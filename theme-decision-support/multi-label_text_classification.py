from database import Db, output_datasets
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def learning(dataset):
    def encode_labels(labels, label2index):
        """ラベルをエンコード（ワンホットエンコーディング）する関数"""
        encoded = []
        for label_set in labels:
            vec = [0] * len(label2index)
            for label in label_set:
                vec[label2index[label]] = 1
            encoded.append(vec)
        return encoded


    texts = list(dataset.keys())
    raw_labels = list(dataset.values())

    # 一意なラベルを列挙し、それぞれに一意な整数を割り当てる
    all_labels = set(label for sublist in raw_labels for label in sublist)
    label2index = {label: idx for idx, label in enumerate(all_labels)}

    # ラベルをベクトル化
    labels = encode_labels(raw_labels, label2index)


    # トークナイザの初期化
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

    # テキストをトークン化
    input_ids = [tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True) for text in texts]

    # 最大のシーケンス長を取得
    max_length = max([len(ids) for ids in input_ids])

    # 各シーケンスにパディングを追加
    padded_input_ids = []
    for ids in input_ids:
        padded_input_ids.append(ids + [0] * (max_length - len(ids)))

    # torch.Tensorに変換
    input_ids = torch.tensor(padded_input_ids)

    attention_masks = [[1] * len(input_id) + [0] * (max_length - len(input_id)) for input_id in input_ids]
    attention_masks = torch.tensor(attention_masks)

    labels = torch.tensor(labels)

    # DataLoaderの作成
    batch_size = 8
    data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    # GPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの初期化
    num_labels = len(labels[0])
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=num_labels)
    model = model.to(device)
    model.train()

    # 最適化アルゴリズムと損失関数の設定
    optimizer = Adam(model.parameters(), lr=2e-5)
    loss_fn = BCEWithLogitsLoss()

    # 訓練ループ
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)  # バッチのデータをGPUに移動
            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            loss = loss_fn(logits, b_labels.type_as(logits))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch: {epoch}, Loss: {total_loss / len(dataloader)}")

    print("Training complete.")


def main():
    datasets = {}
    data = output_datasets()
    
    for d in data:
        datasets[d[1]] = d[6].split(",")
        
    learning(datasets)



if __name__ == "__main__":
    main()