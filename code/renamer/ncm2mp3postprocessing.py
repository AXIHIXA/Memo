import os


def main():
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            artists, title = filename.split(' - ')
            print(f'{filename} => {title}')
            os.rename(filename, title)
    

if __name__ == '__main__':
    main()
