import os 


def main():
    art_over_10 = set()
    art_over_100 = set()
    
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            new_filename = '93558633_p' + filename
            print(filename, '=>', new_filename)
            os.rename(filename, new_filename)


if __name__ == '__main__': 
    main()
