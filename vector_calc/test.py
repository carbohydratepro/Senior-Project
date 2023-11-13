import torch
from transformers import BertTokenizer, BertForMaskedLM

# トークナイザーとモデルのロード
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# GPUが利用可能かどうかを確認し、可能であればデバイスをGPUに設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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

    # MASKされたトークンの予測結果の上位5つを取得
    top_5_tokens = logits[0, mask_token_index, :].topk(10)

    # 予測されたトークンをデコードしてリストに追加
    predicted_tokens = []
    for token_id in top_5_tokens.indices[0]:
        predicted_tokens.append(tokenizer.decode([token_id]))

    return predicted_tokens

def main():
    # 予測したい文章を用意 (MASKはトークナイザーのmask_tokenで置き換える)
    sentence = "[MASK]を用いた観光情報提示Androidアプリケーション"

    # MASKされたトークンを予測
    predicted_tokens = predict_masked_token(sentence)

    print(f"The top 5 predicted words are: {predicted_tokens}")

if __name__ == "__main__":
    main()
