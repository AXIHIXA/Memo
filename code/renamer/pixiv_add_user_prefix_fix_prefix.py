import os
import sys 


def fix_user_prefix(tgt_dir, src_id, dst_id):
    for filename in os.listdir(tgt_dir):
        if os.path.isfile(filename):
            if filename.startswith(f'u{src_id}'):
                src = os.path.join(tgt_dir, filename)
                dst = src.replace(f'u{src_id}', f'u{dst_id}')
                print(f'{src} -> {dst}')
                os.rename(src, dst)


def main():
    tgt_dir = '.'
    src_id = int(sys.argv[1])
    dst_id = int(sys.argv[2])
    fix_user_prefix(tgt_dir, src_id, dst_id)
    

if __name__ == '__main__': 
    main()
