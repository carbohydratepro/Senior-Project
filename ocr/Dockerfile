# docker build -t my-pyocr-app .
# docker run -it --rm my-pyocr-app

# ocr.pyに変更があった場合はbuildし直し

# 基本イメージとしてPythonの公式イメージを使用
FROM python:3.8

# Tesseract OCRのインストール
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-jpn

# pyocrのインストール
RUN pip install pyocr pillow

# 作業ディレクトリの設定
WORKDIR /app

# 必要なファイルをコンテナにコピー
COPY . /app

# コンテナ起動時に実行するコマンド
CMD ["python", "ocr.py"]

