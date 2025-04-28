import os
import shutil


def main():
    replace_pairs = [('艾尔赛', '艾尔泽'),
                     ('艾尔洁', '艾尔泽'),
                     ('爱尔洁', '艾尔泽'),
                     ('琳赛', '琳泽'),
                     ('琳洁', '琳泽'),
                     ('由美娜', '尤米娜'),
                     ('莉恩', '翎'),
                     ('逸仙', '伊神武'), 
                     ('火来也', '火魔法'),
                     ('土来也', '土魔法'),
                     ('水来也', '水魔法'),
                     ('风来也', '风魔法'),
                     ('光来也', '光魔法'),
                     ('哀家', '妾身'), 
                     ('呗', ''),
                     ('是也', '')]

    for dirpath, dirname, filenames in os.walk('.'):
        for filename in filenames:
            if filename[-4:] == 'epub':
                new_filename = filename[:-4] + 'zip'
                os.system(f'mv \"{filename}\" \"{new_filename}\"')
                os.system(f'unzip -q \"{new_filename}\" -d \"{new_filename[:-4]}\"')
                os.system(f'rm \"{new_filename}\"')
            
    for dirpath, dirname, filenames in os.walk('.'):
        for filename in filenames:
            if filename[-5:] == 'xhtml':
                with open(os.path.join(dirpath, filename), 'r') as fin:
                    content = fin.read()
                    for s, t in replace_pairs:
                        content = content.replace(s, t)
                with open(os.path.join(dirpath, filename), 'w') as fout:
                    fout.write(content)

    for dirname in os.listdir('.'):
        shutil.make_archive(dirname, 'zip', dirname)
        os.system(f'mv \"{dirname}.zip\" \"{dirname}.epub\"')
        os.system(f'rm -r \"{dirname}\"')


if __name__ == '__main__':
    main()
