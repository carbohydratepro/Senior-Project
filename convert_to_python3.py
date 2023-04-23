import os
import subprocess

def convert_directory_to_python3(directory, backup_extension='.bak'):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                cmd = [
                    '2to3', '-W', '--add-suffix={}'.format(backup_extension),
                    '-n', '-W', '-f', 'all', file_path
                ]
                subprocess.run(cmd, check=True)

if __name__ == '__main__':
    target_directory = './deepcoder-master'
    convert_directory_to_python3(target_directory)