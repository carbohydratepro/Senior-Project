import os
import glob
import re
import pandas as pd
from pdfminer.high_level import extract_text
from tqdm import tqdm
from retext import return_re_info


def find_files(directory, extension):
    """ディレクトリのパス, 取得したい拡張子"""
    # ディレクトリのパスを正規化し、末尾に/を追加（Windowsの場合は\）
    directory = os.path.normpath(directory) + os.sep
    
    # globを使って指定した拡張子を持つファイルのパスを全て取得
    return glob.glob(directory + '**/*.' + extension, recursive=True)

def remove_spaces(text):
    # 空白文字を削除
    return text.replace(" ", "")

def convert_pdf_to_txt(file_path):
    # PDFからテキストを抽出
    text = extract_text(file_path)
    return text

def extract_info(text, regex):
    # 情報を抽出
    info_match = re.search(regex, text)
    info = info_match.group(1) if info_match else None
    try:
        info = info.replace('\n', '')
    except:
        pass

    if info == None:
        return regex

    return info




def main():
    thesis_informations = return_re_info() #論文の正規表現を取得
    pdf_informations = []

    # 年度毎にPDFファイルを処理
    for informations in thesis_informations:
        regex = [info for info in informations]
        # 年度, 学部, 学科, 名前, 学籍番号, 提出日, タイトル
        year, faculty, department, name, student_id, submission_date, title = informations

        # 走査するディレクトリのパス
        pdf_paths = find_files(f"../卒論PDF/{year}", "pdf")

        # 卒論PDF配下の年度dirを順番に処理
        for path in tqdm(pdf_paths):
            pdf_text = convert_pdf_to_txt(path)
            pdf_text = remove_spaces(pdf_text)

            for i, info in enumerate(regex):
                try:
                   informations[i] = extract_info(pdf_text, info)
                except:
                    pass 
            # 年度, 学部, 学科, 名前, 学籍番号, 提出日, タイトル
            year, faculty, department, name, student_id, submission_date, title = informations
            path = path.replace(f"..\\卒論PDF\\{year}\\", '')
            pdf_informations.append([year, faculty, department, name, student_id, submission_date, path, title])

    labels = ['year', 'faculty', 'department', 'name', 'student_id', 'submission_date', 'path', 'title']

    # pandasのデータフレームに変換
    df = pd.DataFrame(pdf_informations, columns=labels)

    # CSVファイルに出力
    df.to_csv('./gpt-suggest/output.csv', index=False)


if __name__ == "__main__":
    main()