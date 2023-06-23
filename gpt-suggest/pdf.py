from pdfminer.high_level import extract_text

def convert_pdf_to_txt(file_path):
    # PDFからテキストを抽出
    text = extract_text(file_path)
    return text


def main():
    file_path = f"filepath" # PDFファイルのパスを指定
    text = convert_pdf_to_txt(file_path)
    print(text)


if __name__ == "__main__":
    main()