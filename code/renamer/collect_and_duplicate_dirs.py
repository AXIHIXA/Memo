import os


WHERE_TO_COLLECT = '/mnt/g/HfHf'

WHERE_TO_DUPLICATE = '.'


def collect_and_duplicate():
    for dirpath, dirnames, filenames in os.walk(WHERE_TO_COLLECT):
        for dirname in dirnames: 
            src_dir = os.path.join(dirpath, dirname)
            tgt_dir = os.path.join(dirpath.replace(WHERE_TO_COLLECT, WHERE_TO_DUPLICATE), dirname)
            os.makedirs(tgt_dir)  # Recursive.
            print(f'{src_dir} => {tgt_dir}')



def main() -> None:
    collect_and_duplicate()


if __name__ == '__main__':
    main()
