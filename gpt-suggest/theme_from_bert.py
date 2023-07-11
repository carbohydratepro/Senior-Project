from transformers import BertTokenizer, BertForSequenceClassification
import torch

# モデルとトークナイザーのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 入力テキストの準備
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# モデルの実行
labels = torch.tensor([1]).unsqueeze(0)  # バッチサイズ 1
outputs = model(**inputs, labels=labels)

loss = outputs.loss
logits = outputs.logits
