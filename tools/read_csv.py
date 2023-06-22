import os
import csv
from tqdm import tqdm

def read_csv_files(directory, num_files, num_lines):
    data_set = []
    count_files = 0
    for file_name in tqdm(os.listdir(directory), postfix="csvファイルから情報を取得中"):
        if count_files >= num_files:
            break

        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            with open(file_path, mode='r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                count_lines = 0
                for row in csv_reader:
                    if count_lines >= num_lines:
                        break
                    data_set.append(row)
                    count_lines += 1
            count_files += 1
    return data_set


def read_csv_info(file_path, num_lines):
    info = []
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        count_lines = 0
        # 最初の一行は処理を行わない
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            if count_lines >= num_lines:
                break
            info.append(row)
            count_lines += 1

    return info


def data_output(data):
    for d in data:
        print(d)

def main():
    print("details")


if __name__ == '__main__':
    # dir_path = "./syntax-analysis/Project_CodeNet/metadata"
    # num_files = 10
    # num_lines = 10
    # data_set = read_csv_files(dir_path, num_files, num_lines)
    # data_output(data_set)
    main()
