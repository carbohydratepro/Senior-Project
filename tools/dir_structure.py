import os


def load_ignore_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def print_dir_structure(startpath, ignore_list):
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in ignore_list]
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        sub_indent = ' ' * 4 * (level + 1)

        # Display directories first
        for d in dirs:
            print('{}[D] {}'.format(sub_indent, d))

        # Ignore hidden files and display the rest
        for f in files:
            if not f.startswith('.'):
                print('{}[F] {}'.format(sub_indent, f))

ignore_list = load_ignore_list('./tools/ignore.txt')
print_dir_structure('./', ignore_list)

