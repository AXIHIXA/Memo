import os
import shutil


def rename():
    post_id: str = '46171285'

    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            num, suffix = filename.split('.')
            num = int(num)
            new_filename: str = '{}_p{:02d}.{}'.format(post_id, num, suffix)
            print(filename, '==>', new_filename)
            shutil.move(filename, new_filename)


def combine():
    page = 1

    for a, b in zip(os.listdir('../a'), reversed(os.listdir('../b'))):
        shutil.copyfile(os.path.join('../a', a), '{:03d}.jpg'.format(page))
        page += 1
        shutil.copyfile(os.path.join('../b', b), '{:03d}.jpg'.format(page))
        page += 1


def main():
    rename()
    # combine()


if __name__ == '__main__':
    main()
