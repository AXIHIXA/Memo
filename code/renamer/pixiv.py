import os 


def main():
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            if '.gif' in filename and '_p' not in filename:
                    prefix = filename.split('.')[0]
                    new_filename = prefix + '_p0.gif'
                    print(filename, '=>', new_filename)
                    os.rename(filename, new_filename)
    
    art_size = {}

    for root, dirs, files in os.walk('./'):
        for filename in files:
            if '.py' not in filename:
                try:
                    prefix, lst = filename.split('_p')
                    num, suffix = lst.split('.')
                    num = int(num)

                    if prefix in art_size:
                        art_size[prefix] = max(art_size[prefix], num)
                    else:
                        art_size[prefix] = num
                
                except Exception as e:
                    print(filename)
                    raise e
    
    for root, dirs, files in os.walk('./'):
        for filename in files:
            if '.py' not in filename:
                prefix, lst = filename.split('_p')

                if art_size[prefix] < 10:
                    continue

                num, suffix = lst.split('.')
                num = int(num)

                if art_size[prefix] < 100:
                    new_filename = '{}_p{:02d}.{}'.format(prefix, num, suffix)
                else:  # 100 <= art_size[prefix]
                    new_filename = '{}_p{:03d}.{}'.format(prefix, num, suffix)

                print(filename, '=>', new_filename)
                os.rename(filename, new_filename)


if __name__ == '__main__': 
    main()
