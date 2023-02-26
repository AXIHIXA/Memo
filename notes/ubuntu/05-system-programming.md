# Linux System Programming



### 🌱 Commonly-used Commands

#### 🎯 Terminal

- `Ctrl + Shift + F1`: GUI
- `Ctrl + Shift + F2-FX`: TTY terminal
- TTY: teletype (black screen, no GUI)
- PTS: GUI terminal @ `/dev/pts/x`

#### 🎯 `w` & `tty`

- `w`: See all login-ed users, does not show remote sessions (e.g., vscode ssh, has no login at all)
```
$ w
 20:21:57 up 8 days, 20:36,  1 user,  load average: 0.47, 0.33, 0.18
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
xihan1   :1       :1               16Feb23 ?xdm?  14:21m  0.00s /usr/lib/gdm3/gdm-x-session --run-script env GNOME_SHELL_SESSION_MODE=ubuntu /usr/b
```
- `tty`: See current terminal
```
$ tty
/dev/pts/1
```

#### 🎯 `df`

- See hardware mount points
```
$ df
Filesystem     1K-blocks      Used Available Use% Mounted on
udev            32849452         0  32849452   0% /dev
tmpfs            6577592      2472   6575120   1% /run
/dev/nvme0n1p7 946248720 336514664 561593512  38% /
tmpfs           32887952     66904  32821048   1% /dev/shm
tmpfs               5120         4      5116   1% /run/lock
tmpfs           32887952         0  32887952   0% /sys/fs/cgroup
/dev/loop0           128       128         0 100% /snap/bare/5
/dev/loop2         56960     56960         0 100% /snap/core18/2679
/dev/loop3        168832    168832         0 100% /snap/gnome-3-28-1804/161
/dev/loop4        166784    166784         0 100% /snap/gnome-3-28-1804/145
/dev/loop5         64896     64896         0 100% /snap/core20/1822
/dev/loop6         47104     47104         0 100% /snap/snap-store/638
/dev/loop7         47104     47104         0 100% /snap/snap-store/599
/dev/loop12        51072     51072         0 100% /snap/snapd/18357
/dev/loop8         64896     64896         0 100% /snap/core20/1778
/dev/loop11       224256    224256         0 100% /snap/gnome-3-34-1804/77
/dev/loop15       354688    354688         0 100% /snap/gnome-3-38-2004/119
/dev/loop16        93952     93952         0 100% /snap/gtk-common-themes/1535
/dev/loop9        224256    224256         0 100% /snap/gnome-3-34-1804/72
/dev/loop13        51072     51072         0 100% /snap/snapd/17950
/dev/loop10        83328     83328         0 100% /snap/gtk-common-themes/1534
/dev/loop14       354688    354688         0 100% /snap/gnome-3-38-2004/115
/dev/nvme0n1p1    149504     91386     58118  62% /boot/efi
tmpfs            6577588        20   6577568   1% /run/user/125
tmpfs            6577588        72   6577516   1% /run/user/1002
/dev/loop17        56960     56960         0 100% /snap/core18/2697
```

#### 🎯 `man`

- `man XXXX`: See manual page of XXXX. 
- `man man`: See manual page of `man` command. 
- Inside man window: 
  - `h`: Help page
  - `/pattern`: Search forward
  - `?pattern`: Search backward
  - `n`: See next search result
  - `N`: See previous search result
- Sections (A same name may has multiple entries/usages in multiple sections, e.g., `stat`): 
  - `man stat`: See first `stat` (which is in section 1)
  - `man 2 stat`: See `stat` in section 2. 
```
1   Executable programs or shell commands
2   System calls (functions provided by the kernel)
3   Library calls (functions within program libraries)
```

#### 🎯 Redirection & Pipe

- `ls >> out.txt`: Append stdout to `out.txt`
- `ls > out.txt`: Overwrite `out.txt`with stdout
- `ls N> file`: `N` default to 1 (stdout). 
- File descriptors: 
  - stdin: 0
  - stdout: 1
  - stderr: 2



### 🌱 File I/O

#### 🎯 File Contents

- These system calls are used for reading/writing sockets, etc. 
- For regular files, go to C/C++ library functions (with optimizations such as buffers). 
- `open`, `read`, `close`
```c++
#include <iostream>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/unistd.h>

int main(int argc, char * argv[])
{
    if (int fd = open("var/in/1.txt", O_RDONLY); fd != -1)
    {
        ssize_t charsRead = 0L;
        std::string buf(8UL, '\0');

        while ((charsRead = read(fd, buf.data(), 8UL)))
        {
            if (charsRead == -1L)
            {
                std::perror("read func error");
                return EXIT_FAILURE;
            }

            std::cout << buf << '\n';
        }

        close(fd);
    }
    else
    {
        std::perror("failed to open file");
        close(fd);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```
- `open`, `write`, `close`
```c++
if (int fd = open("var/out/1.txt", O_WRONLY | O_CREAT | O_APPEND, 0644); fd != -1)
{
    std::string buf = "1234567890\n";
    
    if (write(fd, buf.c_str(), buf.size()) == -1L)
    {
        perror("write func error");
        return EXIT_FAILURE;
    }

    close(fd);
}
else
{
    std::perror("failed to open file");
    close(fd);
    return EXIT_FAILURE;
}
```
- `__off_t ret = lseek(FD, OFFSET, WHENCE);`
  - Move `fd`'s file position to `OFFSET` bytes from 
    - the beginning of the file (if `WHENCE` is `SEEK_SET`),
    - the current position (if `WHENCE` is `SEEK_CUR`),
    - the end of the file (if `WHENCE` is `SEEK_END`).
  - Return the new file position.  
  - Frequently used for driver programming. 

#### 🎯 File Attributes

- `stat` (queries actual file attributes if link), `lstat` (queries link file for links)
```c++
#include <iostream>
#include <sys/stat.h>

int main(int argc, char * argv[])
{
    if (struct stat statBuf {}; stat("var/in/1.txt", &statBuf) != -1)
    {
        std::cout << statBuf.st_size << '\n';
    }
    else
    {
        perror("stat func error");
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
```
- `opendir`, `readdir`, `closedir`
```c++
#include <iostream>
#include <dirent.h>

void walk(const std::string & path)
{
    if (DIR * dir = opendir(path.c_str()); dir)
    {
        dirent * pDirent;
        int oldErrno = errno;

        while ((pDirent = readdir(dir)))
        {
            std::string name = pDirent->d_name;

            if (name == "." || name == "..")
            {
                continue;
            }

            if (pDirent->d_type == DT_DIR)
            {
                walk(path + '/' + std::string(pDirent->d_name));
            }
            else
            {
                std::cout << path + '/' + std::string(pDirent->d_name) << '\n';
            }
        }

        if (errno != oldErrno)
        {
            perror("readdir error");
            return;
        }

        closedir(dir);
    }
    else
    {
        perror("opendir error");
        closedir(dir);
        return;
    }
}
```

#### 🎯 File Descriptors

- `dup`, `dup2`, `dup3`
```c
/* Duplicate FD, returning a new file descriptor on the same file.  */
extern int dup (int __fd) __THROW __wur;

/* Duplicate FD to FD2, closing FD2 and making it open on the same file.  */
extern int dup2 (int __fd, int __fd2) __THROW;

/* Duplicate FD to FD2, closing FD2 and making it open on the same
   file while setting flags according to FLAGS.  */
extern int dup3 (int __fd, int __fd2, int __flags) __THROW;
```
```c++

```














### 🌱 

#### 🎯 

##### 📌 

### 🌱 

#### 🎯 

##### 📌 

### 🌱 

#### 🎯 

##### 📌 
