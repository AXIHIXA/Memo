import os


def kemono() -> None:
    rnmp: dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            postid: str = dirpath.split(' ')[1][1:-1]

            lst: list[str] = filename.split('.')
            prefix: str = lst[-2]
            suffix: str = lst[-1]
            picid: str = prefix.split('_')[0]
            
            filename: str = os.path.join(dirpath, filename)
            new_filename: str = os.path.join('./', 'fan_{}_p{}.{}'.format(postid, picid, suffix))

            if new_filename in rnmp:
                raise Exception(f'{filename} => {new_filename}, but {new_filename} already exists')
            else:
                rnmp[new_filename] = filename

    for new_filename, filename in rnmp.items():
        print(f'{filename} => {new_filename}')
        os.rename(filename, new_filename)


def main() -> None:
    kemono()


if __name__ == '__main__':
    main()
