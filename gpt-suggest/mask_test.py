import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.optim import AdamW
from janome.tokenizer import Tokenizer

main_text = """
国内における聴覚・言語障碍者は約34万人いるとされている．手話以外の会話
方法として一般に筆談が用いられることがあるが、道具を必要とする点やリアルタ
イム性に欠ける点を改善することは難しい．さらに遠隔でカメラ越しでの会話が普
及し，他言語からの自動翻訳が求められている．その一方で手話の自動翻訳は普及
が進まず，一般的に利用されているとはいえない．

本研究では被識別者（手話話者）を正面から写すことのできる，1台のカメラに
よる動画の手話識別を目的としてシステムの検討を行う．提案システムの特徴とし
て関節軌跡情報の透過率を変化させることで手話の一連動作を識別可能としている．
また，関節情報のみを用いることで，話者の服装や背景などの環境に左右されない
システムとなっている．

提案システムの流れとして，一連の手話動作画像からOpenPoseを用いて関節
座標を取得する．その座標から軌跡として画像に出力したものを学習データとする．
生成した画像の分類はCNNを利用したモデルで行う.本提案では一連の動作の中
で動きのあるものを特徴として抽出できる点，色情報として時間方向の情報を保持
できる点が挙げられる．さらに画像識別において不要な情報を排除することができ，
識別率の向上が見込める．本研究ではその画像を左右の手と全身の関節画像に分け
てCNNモデルに入力することで，11種類のクラス分類について学習データと同じ
環境で95%，一連の手話動画による検証では87.8%の高い精度を得ることができた．
また，学習に用いなかった他被験者3人での手話識別についても80.1%の識別率が
得られた．このことから一人の手話を学習させたモデルにおいて他者での識別も可
能であると示すことができた．今後は学習データを拡充させ，さらに多くのクラス
でも手話識別が可能であるか検証を行う必要がある．
"""


def analyze_text(text):
    t = Tokenizer()
    
    # 結果を保存するためのリスト
    tokens_list = []
    tokens_with_pos_dict = {}
    
    tokens = list(t.tokenize(text))
    
    # 名詞の結合処理
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.part_of_speech.split(',')[0] == '名詞' and i+1 < len(tokens) and tokens[i+1].part_of_speech.split(',')[0] == '名詞':
            compound_noun = token.surface
            while i+1 < len(tokens) and tokens[i+1].part_of_speech.split(',')[0] == '名詞':
                i += 1
                compound_noun += tokens[i].surface
            tokens_list.append(compound_noun)
            tokens_with_pos_dict[compound_noun] = '名詞'
        else:
            tokens_list.append(token.surface)
            tokens_with_pos_dict[token.surface] = token.part_of_speech.split(',')[0]
        i += 1
        
    return tokens_list, tokens_with_pos_dict

def main():
    tokens, tokens_with_pos = analyze_text(main_text)
    model_name = "cl-tohoku/bert-large-japanese"
    unmasker = pipeline('fill-mask', model=model_name)

    for text, pos in tokens_with_pos.items():
        if pos == "名詞":
            # textがmain_text中で1回だけ出現するもののみを置き換える
            if main_text.count(text) == 1:
                masked_text = main_text.replace(text, "[MASK]", 1)
                results = unmasker(masked_text)

                # resultsが辞書の場合、リストとして処理
                if isinstance(results, dict):
                    results = [results]

                print(text)
                for result in results:
                    if isinstance(result, dict) and "token_str" in result:
                        token_str = result["token_str"]
                        score = result["score"]
                        print(f"{token_str}:{score:.5f}")

if __name__ == "__main__":
    main()
    


