import os

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, error


def main():
    for dirpath, dirnames, filenames in os.walk('./'):
        for filename in filenames:
            audio = MP3(os.path.join(dirpath, filename), ID3=ID3)
            
            try:
                audio.add_tags()
            except error:
                pass
            
            audio.tags.add(APIC(mime='image/jpeg', type=3, desc=u'Cover', data=open('../c4ca4238a0b923820dcc509a6f75849b-84.jpg', 'rb').read()))
            audio.save()
    

if __name__ == '__main__':
    main()
