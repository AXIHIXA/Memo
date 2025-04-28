import os 


def main():
    art_over_10 = set()
    art_over_100 = set()
    
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            if '.gif' in filename and '_p' not in filename:
                    prefix = filename.split('.')[0]
                    new_filename = prefix + '_p0.gif'
                    print(filename, '=>', new_filename)
                    os.rename(filename, new_filename)
    
    for root, dirs, files in os.walk('./'):
        for filename in files:
            if '.py' not in filename:
                try:
                    prefix, lst = filename.split('_p')
                    num, suffix = lst.split('.')
                    num = int(num)
                    
                    if num > 9:
                        art_over_10.add(prefix)
                        
                    if num > 99: 
                        art_over_100.add(prefix)
                
                except Exception as e:
                    print(filename)
                    raise e
    
    for root, dirs, files in os.walk('./'):
        for filename in files:
            if '.py' not in filename:
                prefix, lst = filename.split('_p')
                
                if prefix in art_over_10:
                    num, suffix = lst.split('.')
                    num = int(num)
                    new_filename = '{}_p{:02d}.{}'.format(prefix, num, suffix)
                    print(filename, '=>', new_filename)
                    os.rename(filename, new_filename)
                    
                if prefix in art_over_100:
                    num, suffix = lst.split('.')
                    num = int(num)
                    new_filename = '{}_p{:03d}.{}'.format(prefix, num, suffix)
                    print(filename, '=>', new_filename)
                    os.rename(filename, new_filename)


if __name__ == '__main__': 
    main()
