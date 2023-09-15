from database import output_datasets
from transformers import BertTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers.models.bert import BertPreTrainedModel, BertModel
from sklearn.metrics import jaccard_score, hamming_loss
import torch.nn as nn
import numpy as np
import torch
import logging
import random
import re

# ログの設定
logging.basicConfig(level=logging.INFO)

# mlb = MultiLabelBinarizer()


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None):
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs
        
        return outputs  # (loss, logits, hidden_states, attentions)

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)



def compute_jaccard_index(predictions, true_labels):
    return jaccard_score(true_labels, predictions, average='samples')

def compute_hamming_loss(predictions, true_labels):
    return hamming_loss(true_labels, predictions)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    
    return {
        'jaccard_index': compute_jaccard_index(predictions, labels),
        'hamming_loss': compute_hamming_loss(predictions, labels),
        'accuracy': accuracy_score(labels, predictions)
    }

def load_tokenizer_and_model(num_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return tokenizer, model

def preprocess_data(data, tokenizer):
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    mlb_local = MultiLabelBinarizer()
    labels = mlb_local.fit_transform(labels)
    num_labels = len(mlb_local.classes_)
    
    def tokenize_function(examples):
        return tokenizer(examples, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    
    encodings = tokenize_function(texts)
    return encodings, labels, num_labels, mlb_local  # mlb_localを返す

def split_data(encodings, labels):
    # 辞書の各キーに対してtrain_test_splitを適用
    train_data = {}
    test_data = {}
    for key, value in encodings.items():
        train_data[key], test_data[key], _, _ = train_test_split(value, labels, test_size=0.2)

    train_labels, test_labels = train_test_split(labels, test_size=0.2)

    return train_data, test_data, torch.tensor(train_labels), torch.tensor(test_labels)

def train_model(train_encodings, train_labels, test_encodings, test_labels, model):
    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir='./theme-decision-support/logs',
        logging_steps=100,
        do_train=True,
        do_eval=True,
        output_dir='./theme-decision-support/results',
        overwrite_output_dir=True,
        save_steps=100,
        dataloader_pin_memory=False,
        eval_steps=100,
        save_total_limit=2,
    )
    # データをカスタムデータセットにラップ
    train_dataset = CustomDataset(train_encodings, train_labels)

    eval_dataset = CustomDataset(test_encodings, test_labels)

    # Trainerのインスタンス化の際に、このカスタムデータセットを使用
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # これを追加
        compute_metrics=compute_metrics,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                    'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                    'labels': torch.stack([f['labels'] for f in data])}
    )

    trainer.train()
    
    # モデルの保存
    model.save_pretrained("./theme-decision-support/model")




# 予測の取得
def get_predictions(encodings, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for idx in range(encodings['input_ids'].shape[0]):
            input_ids = encodings['input_ids'][idx].unsqueeze(0).to(device)
            attention_mask = encodings['attention_mask'][idx].unsqueeze(0).to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_predictions.append(predictions[0])
    return all_predictions

def display_predictions(test_encodings, tokenizer, mlb_local):
    # モデルのロード
    model_path = "./theme-decision-support/model"
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_path)
    
    # テストデータの予測確率を取得
    predicted_probabilities = get_predictions(test_encodings, model)

    # 上位5つのラベルのインデックスを取得
    top5_indices = np.argsort(predicted_probabilities, axis=1)[:, -5:]

    # 上位5つのラベルだけを1に設定し、それ以外を0に設定する新しいバイナリ予測配列を作成
    top5_predictions = np.zeros_like(predicted_probabilities)
    for i, indices in enumerate(top5_indices):
        top5_predictions[i, indices] = 1

    # 予測のデコード
    decoded_predictions = mlb_local.inverse_transform(top5_predictions)

    # テストデータのテキストの取得
    test_texts = [tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in test_encodings['input_ids']]
    
    # 結果の表示
    for idx, (text, pred_labels) in enumerate(zip(test_texts, decoded_predictions)):
        print(f"Text: {text}")
        print(f"Predicted Labels: {', '.join(pred_labels)}\n")
        if idx == 3:
            break


# テストデータのトークン化
def tokenize_test_texts(test_texts, tokenizer):
    return tokenizer(test_texts, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

def main():
    logging.info("データベースからデータを取得中")
    datasets = []
    data = output_datasets()
    # data = random.sample(data, 100)

    for d in data:
        datasets.append([d[1], re.split('[ ,]', d[6])])
    
    # GPUが利用可能かを確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # まず、トークナイザとモデルを読み込む（ラベルの数が不明なので、仮に2で初期化）
    tokenizer, _ = load_tokenizer_and_model(2)

    encodings, labels, num_labels, mlb_local = preprocess_data(datasets, tokenizer)
    tokenizer, model = load_tokenizer_and_model(num_labels)  # num_labelsを元に再度ロード
    
    train_encodings, test_encodings, train_labels, test_labels = split_data(encodings, labels)
    
    model.to(device)
    train_encodings = {key: val.to(device) for key, val in train_encodings.items()}
    train_labels = train_labels.to(device)
    
    train_model(train_encodings, train_labels, test_encodings, test_labels, model)


    test_texts = [item[0] for item in datasets]  # データセットからテキストの部分だけを抜き出す
    test_encodings = tokenize_test_texts(test_texts, tokenizer)

    # 予測の表示
    display_predictions(test_encodings, tokenizer, mlb_local)

if __name__ == "__main__":
    main()
