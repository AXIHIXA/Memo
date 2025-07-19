import os
import sys 


def main():
    user_id = int(sys.argv[1])

    for dirpath, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            dst = os.path.join(dirpath, f'u{user_id}_{filename}')
            print(f'{src} -> {dst}')
            os.rename(src, dst)


if __name__ == '__main__': 
    main()
