import os
import sys 


def prescreen(target_directory: str, user_id: int) -> bool:
    ok = True

    for filename in os.listdir(target_directory):
        if os.path.isfile(filename):
            if filename[0].isdigit() or filename[0:3] in ['fan', 'epa', 'pat', 'zex']:
                pass
            
            elif filename.startswith(f'u{user_id}'):
                print(f'Warning: File {filename} already prefixes with user_id.')
            
            else:
                ok = False
                print(f'Warning: Irregular filename {filename}.')

    return ok


def add_prefix(target_directory: str, user_id: int):
    for filename in os.listdir(target_directory):
        if os.path.isfile(filename):
            if filename[0].isdigit() or filename[0:3] in ['fan', 'epa', 'pat', 'zex']:
                src = os.path.join(target_directory, filename)
                dst = os.path.join(target_directory, f'u{user_id}_{filename}')
                # Takes too long...
                # print(f'{src} -> {dst}')
                os.rename(src, dst)
    print('Done.')


def main():
    user_id = int(sys.argv[1])

    if prescreen('.', user_id):
        add_prefix('.', user_id)
    else:
        print('Please resolve the errors!')


if __name__ == '__main__': 
    main()
