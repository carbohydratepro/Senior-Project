from PIL import Image
import pyocr
import pyocr.builders

def ocr_image(image_path):
    # 利用可能なOCRツールを取得
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("利用可能なOCRツールが見つかりませんでした。")
        return

    # 利用するOCRツールを選択
    tool = tools[0]
    print("使用するOCRツール: %s" % (tool.get_name()))

    # 画像を開く
    img = Image.open(image_path)

    # 画像からテキストを読み取る
    text = tool.image_to_string(
        img,
        lang="jpn",
        builder=pyocr.builders.TextBuilder()
    )

    return text

# 画像ファイルのパス
image_path = './image/image1.png'

# OCR実行
extracted_text = ocr_image(image_path)
print("読み取ったテキスト:")
print(extracted_text.replace("\n", ""))
