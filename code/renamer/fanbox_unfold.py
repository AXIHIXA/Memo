import os


def fanbox_unfold():
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:    
            filename: str = os.path.join(dirpath, filename)
            suffix: str = filename.split('.')[-1]
            prefix: str = filename[:-len(suffix) - 1]
            
            if not prefix[-1].isdigit():
                raise Exception(f'{filename} does not end with a number')
           
            postid: str = prefix.split('fan_')[-1].split('_p')[0]
            picid: str = ''
            
            for c in reversed(prefix):
                if c.isdigit():
                    picid += c
                else: 
                    break
            
            picid = picid[::-1]
            new_filename: str = os.path.join('./', 'fan_{}_p{}.{}'.format(postid, picid, suffix))
            print(f'{filename} => {new_filename}')
            
            if os.path.exists(new_filename):
                raise Exception(f'{new_filename} already exists, check original namings')
            
            os.rename(filename, new_filename)


def main():
    fanbox_unfold()


if __name__ == '__main__': 
    main()
