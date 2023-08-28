import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)

def load_tokenizer_and_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return tokenizer, model

def preprocess_data(data, tokenizer):
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    num_labels = len(mlb.classes_)
    
    def tokenize_function(examples):
        return tokenizer(examples, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    
    encodings = tokenize_function(texts)
    return encodings, labels, num_labels

def split_data(encodings, labels):
    train_encodings, test_encodings, train_labels, test_labels = train_test_split(encodings, labels, test_size=0.2)
    return train_encodings, test_encodings, torch.tensor(train_labels), torch.tensor(test_labels)

def train_model(train_encodings, train_labels, model, num_labels):
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir='./theme-decision-support/logs',
        logging_steps=10,
        do_train=True,
        do_eval=True,
        output_dir='./theme-decision-support/results',
        overwrite_output_dir=True,
        save_steps=10,
        eval_steps=10,
        save_total_limit=2,
    )

    train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    
    # モデルの保存
    model.save_pretrained("./theme-decision-support/model")
    tokenizer.save_pretrained("./theme-decision-support/model")

def main():
    logging.info("データベースからデータを取得中")
    datasets = {}
    data = output_datasets()
    
    for d in data:
        datasets[d[1]] = d[6].split(",")
    
    # GPUが利用可能かを確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_tokenizer_and_model()
    encodings, labels, num_labels = preprocess_data(data, tokenizer)
    train_encodings, test_encodings, train_labels, test_labels = split_data(encodings, labels)
    
    model.to(device)
    train_encodings = {key: val.to(device) for key, val in train_encodings.items()}
    train_labels = train_labels.to(device)
    
    train_model(train_encodings, train_labels, model, num_labels)

if __name__ == "__main__":
    main()
