import os


def main():
    """
    1.png, 2.png, ..., 74.png => fan_5750202_p19.png, ...
    """
    id_increment: int = 0
    prefix: str = '104704583'

    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames: 
            index, suffix = filename.split('.')
            new_filename: str = prefix + '_p{:d}'.format(int(index) + id_increment) + '.' + suffix
            print(f'{filename} => {new_filename}')
            os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, new_filename))


if __name__ == '__main__':
    main()
