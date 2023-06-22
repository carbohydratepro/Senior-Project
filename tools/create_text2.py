from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text(text):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)

    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0])

    return summary

text = "平面上の異なる 4 点$A (x_a, y_a)$, $B (x_b, y_b)$, $C (x_c, y_c)$, $D(x_d, y_d)$ の座標を読み込んで、それら 4 点を頂点とした四角形 $ABCD$ に凹みがなければ YES、凹みがあれば NO と出力するプログラムを作成してください。"
print(summarize_text(text))
