import os
import csv

def read_csv_files(directory, num_files, num_lines):
    count_files = 0
    for file_name in os.listdir(directory):
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
                    print(row)
                    count_lines += 1
            count_files += 1

if __name__ == '__main__':
    dir_path = "./syntax-analysis/Project_CodeNet/metadata"
    num_files = 10
    num_lines = 10
    read_csv_files(dir_path, num_files, num_lines)
