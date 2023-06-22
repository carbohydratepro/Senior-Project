from transformers import T5Tokenizer, AutoModelForCausalLM

# gpt-2モデルを使用して文章を生成
# 接頭辞（Prefix）
PREFIX_TEXT = "今まで登ったことのある山の標高を記録したデータがあります。このデータを読み込んで、一番高い山と一番低い山の標高差を出力するプログラムを作成してください。を解くには、"

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")


# 推論
input = tokenizer.encode(PREFIX_TEXT, return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=120, num_return_sequences=1)

content = tokenizer.batch_decode(output)[0]

print(content)