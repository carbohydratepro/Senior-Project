from transformers import BertTokenizer, BertForMultipleChoice
import torch
from torch.nn.functional import softmax

# トークナイザとモデルをロードする
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMultipleChoice.from_pretrained('bert-base-uncased')

# 問題と回答のリスト
questions = ["What is the capital of France?", "What is the capital of England?"]
choices = [["Paris", "London", "Berlin"], ["Paris", "London", "Berlin"]]

# データの準備
examples = []
for question, choice in zip(questions, choices):
    # 選択肢ごとに個別にエンコードし、次にそれらをリストとしてまとめる
    inputs = [tokenizer.encode_plus(question, choice_, add_special_tokens=True, max_length=64, padding='max_length', truncation=True, return_tensors='pt') for choice_ in choice]
    
    # エンコードされた選択肢を1つのテンソルにまとめる
    input_ids = torch.cat([i['input_ids'] for i in inputs], dim=0).unsqueeze(0) 
    attention_mask = torch.cat([i['attention_mask'] for i in inputs], dim=0).unsqueeze(0)
    
    # 正解のラベルをセットする
    labels = torch.tensor(0).unsqueeze(0)  

    examples.append(({'input_ids': input_ids, 'attention_mask': attention_mask}, labels))

# モデルの訓練
model.train()
optimizer = torch.optim.Adam(model.parameters())
for inputs, labels in examples:
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# モデルの評価
model.eval()
for inputs, labels in examples:
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        # ロジットを取得し、Softmax関数に通す
        probabilities = softmax(outputs.logits, dim=1)
        print("Probabilities:", probabilities)
