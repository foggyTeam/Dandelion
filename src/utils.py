import os


def colored_print(text: str, color: str = 'grey'):
    if color == 'grey':
        print(f"\033[90m{text}\033[0m")
    elif color == 'r':
        print(f"\033[91m{text}\033[0m")
    elif color == 'g':
        print(f"\033[92m{text}\033[0m")
    elif color == 'y':
        print(f"\033[93m{text}\033[0m")
    elif color == 'm':
        print(f"\033[95m{text}\033[0m")
    elif color == 'c':
        print(f"\033[96m{text}\033[0m")


def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        raise FileNotFoundError('Folder does not contain model\'s params!')
