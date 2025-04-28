import os 


def main():
    art_over_10 = set()
    art_over_100 = set()

    for root, dirs, files in os.walk('./'):
        for filename in files:
            if '.py' not in filename:
                splited_filename = filename.split('-')
                prefix = splited_filename[1] 
                num = splited_filename[3].split('.')[0][3:]
                num = int(num)
                
                if num > 9:
                    art_over_10.add(prefix)
                    
                if num > 99: 
                    art_over_100.add(prefix)
                    
    for root, dirs, files in os.walk('./'):
        for filename in files:
            if '.py' not in filename:
                splited_filename = filename.split('-')
                prefix = splited_filename[1] 
                num = splited_filename[3].split('.')[0][3:]
                num = int(num)
                suffix = splited_filename[3].split('.')[-1]
                
                if prefix in art_over_10:
                    new_filename = '{}_p{02d}.{}'.format(prefix, num, suffix)
                    print(filename, '=>', new_filename)
                    os.rename(filename, new_filename)
                    
                elif prefix in art_over_100:
                    new_filename = '{}_p{03d}.{}'.format(prefix, num, suffix)
                    print(filename, '=>', new_filename)
                    os.rename(filename, new_filename)
                    
                else:
                    new_filename = '{}_p{}.{}'.format(prefix, num, suffix)
                    print(filename, '=>', new_filename)
                    os.rename(filename, new_filename)


if __name__ == '__main__': 
    main()
