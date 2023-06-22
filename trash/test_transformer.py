import torch
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

# トークナイザとモデルのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# トレーニングデータの準備
# contextは問題の背景情報、questionは問題、answersは答え（開始と終了位置）です
# ここでは単純化のため一つの例を使っていますが、実際には大量のデータが必要です
train_data = [{'context': 'The sky is blue.', 
               'question': 'What color is the sky?', 
               'answers': {'answer_start': [11], 'text': ['blue']}}]
               
train_encodings = tokenizer([d['context'] for d in train_data], 
                            [d['question'] for d in train_data], 
                            truncation=True, padding=True)

# 答えの位置をトークン単位でエンコード
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start'][0]))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_start'][0] + len(answers[i]['text'][0])))
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, [d['answers'] for d in train_data])

# PyTorchのデータセットに変換
class QA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = QA_Dataset(train_encodings)

# トレーニング
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
)

trainer.train()

