from transformers import GPT2LMHeadModel, T5Tokenizer

model_name = "rinna/japanese-gpt2-medium"

# 日本語GPT-2モデルとトークナイザーのロード
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_paper_title(prompt):
    # プロンプトに対してGPT-2を使って文章を生成する
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, temperature=0.7, num_return_sequences=1)

    # 生成した文章をデコードする
    output_str = tokenizer.decode(outputs[0])
    
    # 最初のピリオドまでの文字列（つまり、タイトル）を返す
    title = output_str.split("。")[0]
    return title

prompt = "マルウェアの"  # ここに興味のあるトピックやキーワードを入力
print(generate_paper_title(prompt))
