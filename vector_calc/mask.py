import torch
from transformers import BertTokenizer, BertForMaskedLM

# トークナイザーとモデルのロード
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# GPUが利用可能かどうかを確認します。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルをGPUに移動します。
model.to(device)

# MASKされたトークンを予測する関数
def predict_masked_token(sentence):
    # '[MASK]'トークンを含む文をトークナイズ
    tokenized_sentence = tokenizer.encode(sentence, return_tensors='pt')
    tokenized_sentence = tokenized_sentence.to(device)
    
    # MASKされたトークンの位置を探す
    mask_token_index = torch.where(tokenized_sentence == tokenizer.mask_token_id)[1]

    # 予測を行う
    with torch.no_grad():
        outputs = model(tokenized_sentence)
    logits = outputs.logits

    # MASKされたトークンの予測結果
    predicted_token_id = logits[0, mask_token_index].argmax(axis=1)
    predicted_token = tokenizer.decode(predicted_token_id)

    return predicted_token

def main():
    # 予測したい文章を用意 (MASKはトークナイザーのmask_tokenで置き換える)
    sentence = "東京は日本の[MASK]です。"

    # MASKされたトークンを予測
    predicted_token = predict_masked_token(sentence)

    print(f"The predicted word is: {predicted_token}")


if __name__ == "__main__":
    main()