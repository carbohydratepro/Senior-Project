from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
model = AutoModelForMaskedLM.from_pretrained("nlp-waseda/roberta-base-japanese")

sentence = '早稲田 大学 で 自然 言語 処理 を [MASK] する 。' # input should be segmented into words by Juman++ in advance
encoding = tokenizer(sentence, return_tensors='pt')
...
