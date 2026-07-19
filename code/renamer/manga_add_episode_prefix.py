import os
import sys 


def add_prefix(target_directory: str, prefix: str):
    for filename in os.listdir(target_directory):
        if os.path.isfile(filename):
                src = os.path.join(target_directory, filename)
                dst = os.path.join(target_directory, f'{prefix}_{filename}')
                print(f'{src} -> {dst}')
                os.rename(src, dst)
    print('Done.')


def main():
    prefix = sys.argv[1]
    add_prefix('.', prefix)


if __name__ == '__main__': 
    main()
