import os
import re
import shutil


level_character_to_int = {'一':1, '二':2, '三':3, \
                          '1':1, '2':2, '3':3}
counter = {}

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

def main():
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in sorted(filenames, key=natsort):
            level_character = dirpath.split('/')[2].split('档')[0][-1:]
            level = level_character_to_int[level_character]
            year_month = dirpath.split('/')[1]
            suffix = filename.split('.')[-1]
            key = f'{year_month}_d{level}'
            
            if key in counter:
                counter[key] += 1
            else:
                counter[key] = 1
            
            if level != 2:
                new_filename = 'afd_{}_{:02d}.{}'.format(key, counter[key], suffix)
            else:
                new_filename = 'afd_m_{}_{:02d}.{}'.format(key, counter[key], suffix)
            
            print(os.path.join(dirpath, filename), ' => ', new_filename)
            shutil.copy2(os.path.join(dirpath, filename), os.path.join('./', new_filename))
            # os.rename(os.path.join(dirpath, filename), os.path.join('./', new_filename))


if __name__ == '__main__':
    main()
