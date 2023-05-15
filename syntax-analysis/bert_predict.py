import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
text = '私はお金を稼ぐ'
print("Input text : "+text)

tokenized_text = tokenizer.tokenize(text)
print("Tokenized text : ",tokenized_text)

masked_index = 2
tokenized_text[masked_index] = '[MASK]'
print("Masked text : ",tokenized_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print("Indexed tokens : ",indexed_tokens)

tokens_tensor = torch.tensor([indexed_tokens])
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model.eval()

print("Predicting...")
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0][0, masked_index].topk(5)

print("Predictions : ")
for i, index_t in enumerate(predictions.indices):
    index = index_t.item()
    token = tokenizer.convert_ids_to_tokens([index])[0]
    print(i, token)