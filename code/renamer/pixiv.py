import os


SAVE_DIR = '/mnt/d/AX/ETC/MARCH OF THE PENGUINS'
TMP_DIR = '/mnt/g/TMP'


def rename_a_single_creator(creator_dir):
    for dirpath, dirnames, filenames in os.walk(creator_dir):
        for filename in filenames:
            if '.gif' in filename and '_p' not in filename:
                prefix = filename.split('.')[0]
                new_filename = prefix + '_p0.gif'
                print(filename, '=>', new_filename)
                os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, new_filename))
    
    art_size = {}

    for _, dirs, files in os.walk(creator_dir):
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
    
    for _, dirs, files in os.walk(creator_dir):
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
                os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, new_filename))


def rename_all_creators():
    for creator_dir in os.listdir(TMP_DIR):
        print(f'Renaming {creator_dir}...')
        rename_a_single_creator(os.path.join(TMP_DIR, creator_dir))
        print(f'{creator_dir} done.\n')


def move_all_creators():
    cr_3d = set(os.listdir(os.path.join(SAVE_DIR, '3D')))
    cr_cg = set(os.listdir(os.path.join(SAVE_DIR, 'CG')))
    cr_cg_ai = set(os.listdir(os.path.join(SAVE_DIR, 'CG-AI')))
    cr_cg_ai_ns = set(os.listdir(os.path.join(SAVE_DIR, 'CG-AI-NS')))

    os.mkdir(os.path.join(TMP_DIR, '3D'))
    os.mkdir(os.path.join(TMP_DIR, 'CG'))
    os.mkdir(os.path.join(TMP_DIR, 'CG-AI'))
    os.mkdir(os.path.join(TMP_DIR, 'CG-AI-NS'))

    for cr in os.listdir(TMP_DIR):
        if cr in ['3D', 'CG', 'CG-AI', 'CG-AI-NS']:
            continue

        if cr in cr_3d:
            print(f'{cr} => 3D.')
            os.rename(os.path.join(TMP_DIR, cr), os.path.join(TMP_DIR, '3D', cr))
        elif cr in cr_cg:
            print(f'{cr} => CG.')
            os.rename(os.path.join(TMP_DIR, cr), os.path.join(TMP_DIR, 'CG', cr))
        elif cr in cr_cg_ai:
            print(f'{cr} => CG-AI.')
            os.rename(os.path.join(TMP_DIR, cr), os.path.join(TMP_DIR, 'CG-AI', cr))
        elif cr in cr_cg_ai_ns:
            print(f'{cr} => CG-AI-NS.')
            os.rename(os.path.join(TMP_DIR, cr), os.path.join(TMP_DIR, 'CG-AI-NS', cr))
        else:
            print(f'{cr} NOT found.')


def main():
    rename_all_creators()
    move_all_creators()


if __name__ == '__main__':
    main()
