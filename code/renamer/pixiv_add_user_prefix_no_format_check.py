import os 
import sys


def add_prefix(prefix):
    for filename in os.listdir('.'):
        new_filename = f'u{prefix}_{filename}'
        print(f'{filename} -> {new_filename}')
        os.rename(filename, new_filename)


def main():
    prefix = sys.argv[1]
    add_prefix(prefix)


if __name__ == '__main__': 
    main()
