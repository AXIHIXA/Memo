import os 


def main():
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            os.rename(filename, 'pixiv_user_30716447_' + filename)

            # postid, num = filename.split('_')

            # if num[0] == 'p':
            #     num = num[1:]

            # new_filename = f'fan_{postid}_p{num}'
            # print(filename, '=>', new_filename)
            # os.rename(filename, new_filename)


if __name__ == '__main__': 
    main()
