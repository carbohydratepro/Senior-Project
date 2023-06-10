import gzip
import shutil
import tarfile
import os
from tqdm import tqdm

def extract_python_files(file_path, output_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        for member in tqdm(tar.getmembers()):
            if member.isreg() and member.name.endswith('.py'):  # ファイルがPythonファイルかどうか確認
                # 元のディレクトリ構造を保持するための出力ディレクトリのパスを作成
                full_output_path = os.path.join(output_path, os.path.dirname(member.name))

                # 出力ディレクトリが存在しない場合は作成
                os.makedirs(full_output_path, exist_ok=True)

                # Pythonファイルだけを展開
                tar.extract(member, path=full_output_path)



def extract_bin_gz_files(file_path, output_path):
    with gzip.open('.\\ast-label\\GoogleNews-vectors-negative300.bin.gz', 'rb') as f_in:
        with open('.\\ast-label\\GoogleNews-vectors-negative300.bin', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    file_path = f"E:\Project_CodeNet.tar.gz"
    output_path = f"E:\project_code_net_python_all"
    extract_python_files(file_path, output_path)
