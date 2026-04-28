# 环境配置 Environment Configuration


## 🌱 .BASHRC

- *Note*:
  - Environment variables
    - Should **not** be configured in `.bashrc`! It should stay in `.profile`!
    - See the next section.
  - Clear screen: 
    - `cls` is a Windows PowerShell command to clear the terminal.
    - `reset` is a Linux Bash command to reinitialize the terminal, as if it were opened from scratch.
    - `clear` is a Linux Bash command to clear all previous contents.

### ComputeLab Remote Ubuntu

- 1. ~/.bashrc
```bash
ln -s /home/scratch.xihan_coreai ~/scratch.xihan_coreai
mkdir -p /home/scratch.xihan_coreai/workspace
ln -s /home/scratch.xihan_coreai/workspace ~/workspace
vi ~/.bashrc

# Colored prompts
if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi

# Color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# >>> conda initialize >>>
# ...
# <<< conda initialize <<<

# Xi's personal aliases
SCRATCH=`realpath /home/scratch.xihan_coreai`
WORKSPACE=`realpath $SCRATCH/workspace`
alias c3="conda activate py3"
alias d3="conda deactivate"
alias sc="cd $SCRATCH"
alias ws="cd $WORKSPACE"
alias jl="cd $WORKSPACE; jupyter lab"
alias cls="clear"

# Xi's personal scripts
export PATH="$SCRATCH/opt/bin:$PATH"

# TRT
TRT_USER="xihan"
GITLAB_PAT="YOUR GITLAB PERSONAL ACCESS TOKEN GOES HERE"
export GIT_TRT_ROOT="$WORKSPACE/git-trt"
export PATH="$GIT_TRT_ROOT/bin:$PATH"
export MANPATH="$GIT_TRT_ROOT/man:$MANPATH"
export TRT_ROOT="$WORKSPACE/trt"
export TRT_USER
export TRT_GITLAB_API_TOKEN=$GITLAB_PAT
export TRT_CONTAINERS_PATH="$SCRATCH/.trt_containers"

# TRT autocomplete
if [[ -f ~/.git-trt-autocomplete.bash ]]; then
    source ~/.git-trt-autocomplete.bash
fi
```
- 2. ~/.gitconfig
```bash
vi ~/.gitconfig

[trt]
        gitlab-api-token = "YOUR GITLAB PERSONAL ACCESS TOKEN GOES HERE"
        runc-default-args = "--mounts /home/scratch.xihan_coreai/:/home/xihan/scratch.xihan_coreai/,/home/scratch.xihan_coreai/workspace:/home/xihan/workspace,/home/scratch.svc_compute_arch/:/home/xihan/scratch.svc_compute_arch/"
```
- 3. ~/.profile
```bash
vi ~/.profile

# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
	. "$HOME/.bashrc"
    fi
fi
```
- 4. $SCRATCH/opt/scripts (After [Repo Setup](../git-notes/git-notes.md))
```bash
cd $SCRATCH
mkdir -p opt
cd opt
mkdir -p scripts
cd scripts
vi xi-run-cudnn-container.sh

#!/bin/bash
docker \
    run \
    -it \
    --rm \
    --user=151841:30 \
    --volume=/home/xihan:/home/xihan \
    --volume=/home/scratch.xihan_coreai:/home/scratch.xihan_coreai \
    --workdir=/home/xihan \
    --hostname=docker-`hostname` \
    --name=xihan \
    --gpus=all \
    urm.nvidia.com/hw-cudnn-docker/dev:xihan-local

chmod +x xi-run-cudnn-container.sh
```
```bash
vi xi-nsys-stats-report-cuda-gpu-kern-sum.sh

#!/bin/sh
nsys stats --report cuda_gpu_kern_sum $1

chmod +x xi-nsys-stats-report-cuda-gpu-kern-sum.sh
```
```bash
vi xi-cmake-build.sh 

#!/bin/sh

if [ $# -eq 0 ]; then
    echo "Usage: $0 <Debug|Release> [clean]"
    exit 1
fi

# Convert argument to lowercase for case-insensitive comparison
BUILD_TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# Set CMAKE_BUILD_TYPE and build directory based on input
case "$BUILD_TYPE" in
    debug)
        CMAKE_BUILD_TYPE="Debug"
        BUILD_DIR="cmake-build-debug"
        ;;
    release)
        CMAKE_BUILD_TYPE="Release"
        BUILD_DIR="cmake-build-release"
        ;;
    *)
        echo "Error: Invalid build type '$1'"
        echo "Valid options: Debug, Release (case insensitive)"
        exit 1
        ;;
esac

# Check if the clean argument is provided
if [ $# -ge 2 ]; then
    CLEAN_ARG=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    if [ "$CLEAN_ARG" = "clean" ]; then
        if [ -d "$BUILD_DIR" ]; then
            echo "rm -rf '$BUILD_DIR'"
            rm -rf "$BUILD_DIR"
        fi
    else
        echo "Error: Unknown argument '$2'"
        echo "Valid options: clean (optional, case insensitive)"
        exit 1
    fi
fi

# CMake build
if [ -d "$BUILD_DIR" ]; then
    echo "mkdir '$BUILD_DIR'"
    mkdir "$BUILD_DIR"

    if [ $? -ne 0 ]; then
        echo "Error: failed to create build directory '$BUILD_DIR'"
        exit 1
    fi
fi 

echo "cmake -DCMAKE_BUILD_TYPE='$CMAKE_BUILD_TYPE' -B'$BUILD_DIR'"
cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" -B"$BUILD_DIR"

if [ $? -ne 0 ]; then
    echo "Error: cmake generator returned a non-zero value"
    exit 1
fi

echo "cmake --build '$BUILD_DIR' -j"
cmake --build "$BUILD_DIR" -j

if [ $? -ne 0 ]; then
    echo "Error: cmake builder returned a non-zero value"
    exit 1
fi

echo "Done"
exit 0

chmod +x xi-cmake-build.sh 
```
```bash
cd $SCRATCH
cd opt
mkdir -p bin
cd bin
ln -s `realpath ../scripts/xi-run-cudnn-container.sh` rcd
ln -s `realpath ../scripts/xi-nsys-stats-report-cuda-gpu-kern-sum.sh` nss
ln -s `realpath ../scripts/xi-cmake-build.sh` cmk
```

### MacOS

```bash
vi ~/.zshrc

# Colored prompt
if (( $+commands[tput] )) && tput setaf 1 &>/dev/null; then
    export PROMPT="%B%F{green}%n@%m%f%b:%B%F{blue}%~%f%b %# "
fi

# Colored ls
export CLICOLOR=1
export LSCOLORS=ExGxBxDxCxEgEdxbxgxcxd

# Colored grep
if echo "test" | command grep --color=auto "test" &>/dev/null 2>&1; then
    grep() { command grep --color=auto "$@"; }
    fgrep() { command fgrep --color=auto "$@"; }
    egrep() { command egrep --color=auto "$@"; }
    export GREP_COLORS='ms=01;31:mc=01;31:fn=35:ln=32:se=36'
fi

# >>> conda initialize >>>
# ...
# <<< conda initialize <<<

# Xi's personal aliases
WORKSPACE=`realpath ~/workspace`
alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd $WORKSPACE"
alias jl="cd $WORKSPACE; jupyter lab"
alias cls="reset"
```
```bash
vi ~/.vimrc

" ===== Basic Settings =====
syntax on                      " Enable syntax highlighting
filetype plugin indent on      " Enable file type detection
"set number                     " Show line numbers
"set relativenumber             " Relative line numbers
set ruler                      " Show cursor position
set showcmd                    " Show command in status bar
"set showmode                   " Show current mode
"set cursorline                 " Highlight current line

" ===== Colors =====
set t_Co=256                   " Use 256 colors
"set background=dark            " Dark background
"colorscheme desert             " Color scheme

" ===== Indentation =====
set autoindent                 " Auto-indent new lines
set smartindent                " Smart indentation
set tabstop=4                  " Tab width
set shiftwidth=4               " Indent width
set expandtab                  " Use spaces instead of tabs

" ===== Search =====
set hlsearch                   " Highlight search results
set incsearch                  " Incremental search
set ignorecase                 " Ignore case in search
set smartcase                  " Case-sensitive if uppercase present

" ===== Performance =====
set lazyredraw                 " Don't redraw during macros
set ttyfast                    " Faster terminal

" ===== Backup =====
set nobackup                   " No backup files
set noswapfile                 " No swap files
```


### Real Ubuntu System

```bash
vi ~/.bashrc

# >>> conda initialize >>>
# ...
# <<< conda initialize <<<

# Xi's personal aliases
WORKSPACE=`realpath ~/workspace`
alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd $WORKSPACE"
alias jl="cd $WORKSPACE; jupyter lab"
alias cls="reset"
```

### `WSL`

```bash
vi ~/.bashrc

# >>> conda initialize >>>
# ...
# <<< conda initialize <<<

# Xi's personal aliases
WORKSPACE="/mnt/d/workspace"
alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd $WORKSPACE"
alias jl="cd $WORKSPACE; jupyter lab"
alias cls="reset"
alias ss="sudo service ssh --full-restart"
alias pp="c3; python /mnt/d/workspace/Memo/code/renamer/pixiv_single.py"
alias ppp="c3; python /mnt/d/workspace/Memo/code/renamer/pixiv_multiple.py"
alias ppu="c3; python /mnt/d/workspace/Memo/code/renamer/pixiv_add_user_prefix.py"
alias kkpixiv="bash /mnt/d/workspace/Memo/code/dl-bak/PatreonDownloader-AlexCSDev/download-patreon.sh"
```



## 🌱 环境变量 Environment Variables

### Wiki

- 参考文档
  - [Environment Variables - Ubuntu Documentation](https://help.ubuntu.com/community/EnvironmentVariables). 
- 本地环境变量
  - 建议放置于 `$HOME/.profile` 里。
    - 注意，Anaconda 的初始化脚本以及 alias 除外，这两样必须放在 `$HOME/.bashrc` 里！
    - 这几样是只在 Bash 环境里用的！
  - `$HOME/.profile`
    - 用于初始化 GUI 环境的环境变量。
    - 一个使用 Desktop launcher 启动的应用，其环境变量只由 `$HOME/.profile` 初始化，`$HOME/.bashrc` 中的配置**不会**生效！
      - GUI 中打开 Terminal 用 Bash 的情况下，`$HOME/.profile` 和 `$HOME/.bashrc` 都会生效。
    - `$HOME/.profile` 会检测当前环境，如果是 Bash，则也会执行 `$HOME/.bashrc`，因此不要在 `$HOME/.bashrc` 中执行 `$HOME/.profile`，不然会产生死循环！
  - `$HOME/.bashrc`
    - 用于初始化 Bash 环境的环境变量。
    - 一个使用 Bash 启动的应用，其环境变量只由 `$HOME/.bashrc` 初始化，`$HOME/.profile` 中的配置**不会**生效！
    - 但通过 GUI 启动的 Terminal 中，`$HOME/.profile` 中的配置是生效（继承）的！
- 系统环境变量
  - 建议放置于 `/etc/profile.d/*.sh` 里。
  - `/etc/profile`
    - 由 login shell 执行
      - When you log in via a text console, TTY, SSH, or graphical login manager (like GDM/lightDM)， and a login shell is started (like Bash).
      - It is sourced before the user's `~/.bash_profile`, `~/.bash_login`, or `~/.profile` (whichever exists first).
    - 对于 non-login 或 non-interactive 环境，则不会被执行。
    - While `/etc/profile` is often suggested for setting environment variables system-wide, it is a configuration file of the base-files package, so it's **not** appropriate to edit that file directly. Use a file in `/etc/profile.d` instead as shown above. (Files in `/etc/profile.d` are sourced by `/etc/profile`.)
    - **修改后不会立即生效，需要 log out and log back in！**
      - 其他方式，如切换 TTY，都有丢失当前上下文的风险，重新登录才是 best practice！
- `sudo` Caveat:
  - Any variables added to these locations will **not** be reflected when invoking them with a `sudo` command, as `sudo` has a default policy of resetting the Environment and setting a secure path (this behavior is defined in `/etc/sudoers`). 
    - As a workaround, you can use `sudo su` that will provide a shell with root privileges but retaining any modified `PATH` variables. 
    - Alternatively you can setup sudo not to reset certain environment variables by adding some explicit environment settings to keep in `/etc/sudoers`. 
  - `su` v.s. `su -`
    - `su`：
      - substitute user，切换用户，但保留当前的 working directory 以及环境（环境变量等）；
      - 后面不跟随其他命令时，切换到 root。
    - `su -`：
      - 切换到 root，并启用一个全新的 login shell session。
      - 当前工作目录和上下文不会被保留。


### 如何修改 PATH

```bash
export PATH=/path/to/prepend${PATH:+:${PATH}}
```
- Bash 语法：`${FOO:+${BAR}}`
  - 如果变量 FOO 已定义且非空，则展开为 `${BAR}`；否则，展开为空串
- 上述修改方法，相比 `export PATH=/path/to/prepend:${PATH}` 的好处在于：
  - 如果 PATH 为空，则这种简单写法会被展开为 `export PATH=/path/to/prepend:`，相当于 `export PATH=/path/to/prepend:.`，即将当前目录放进了 PATH，会有安全性问题！
  - 上述复杂写法则能够保证 PATH 为空时的安全性。

### `/etc/profile.d/my-env-vars.sh`

- `/etc/profile.d/*.sh` 会自动被 `/etc/profile` 执行
```bash
sudo touch /etc/profile.d/my-env-vars.sh
sudoedit /etc/profile.d/my-env-vars.sh

# The following are the contents to be appended to /etc/profiles.d/my-env-vars.sh: 

# Tex Live
# $HOME/.bashrc is NOT sourced by GUI apps!
# Set PATH in /etc/profile.d/my-env-vars.sh,
# so that GUI-launched Tex Studio could find correct Tex Live installation. 
export PATH="/usr/local/texlive/2025/bin/x86_64-linux${PATH:+:${PATH}}"

# cuda
export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# editor
export EDITOR="/usr/bin/vim"

# jvm
export JAVA_HOME="/opt/jdk"
export JRE_HOME="${JAVA_HOME}/jre"
export CLASSPATH=".:${JAVA_HOME}/lib:${JRE_HOME}/lib:${CLASSPATH}"
export PATH="${JAVA_HOME}/bin:${JRE_HOME}/bin${PATH:+:${PATH}}"
```



## 🌱 Terminal 

### Editor

- Do NOT use Graphical editors under root privilege, as it could corrupt the files. 
    - Do NOT do this: `sudo gedit ...`
    - Use this command: `sudoedit ...` or `sudo -e ...` 

### Shortcuts

- `Ctrl UKAE`
- `Ctrl` + `u`: Delete all text from cursor to head
- `Ctrl` + `k`: Delete all text from cursor to tail
- `Ctrl` + `a`: Move cursor to head
- `Ctrl` + `e`: Move cursor to end
- `Ctrl` + `w`: Delete one word ahead of cursor
- `Alt` + `f`: Move cursor forward (towards head) by one word
- `Alt` + `b`: Move cursor back (towards tail) by one word



## 🌱 `apt` PPA Management

- Remove PPAs:
    - Use the --remove flag, similar to how the PPA was added:
    ```bash
    sudo add-apt-repository --remove ppa:whatever/ppa
    ```
    - You can also remove PPAs by deleting the `.list` files from `/etc/apt/sources.list.d` directory.
    - As a safer alternative, you can install `ppa-purge`:
    ```bash
    sudo apt-get install ppa-purge
    ```
    - And then remove the PPA, downgrading gracefully packages it provided to packages provided by official repositories:
    ```bash
    sudo ppa-purge ppa:whatever/ppa
    ```
    - Note that this will uninstall packages provided by the PPA, 
      but not those provided by the official repositories. 
      If you want to remove them, you should tell it to `apt`:
    ```bash
    sudo apt-get purge package_name
    ```



## 🌱 Python

### anaconda

- **Best Practive**:
  - Use conda to manage environments ONLY.
  - Use PyPI (pip) to manage packages!
    - PyPI has more updates and more support (e.g., PyTorch no longer supports conda installation);
    - PyPI is much faster than conda to resolve package dependencies!
- Installation:
```
# `bash` is CRITICAL!!! or you may fail when choosing to modify .bashrc
bash ./Anaconda...
```
- Choose `yes`:
```bash
You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes
no change     /home/xihan1/opt/anaconda3/condabin/conda
no change     /home/xihan1/opt/anaconda3/bin/conda
no change     /home/xihan1/opt/anaconda3/bin/conda-env
no change     /home/xihan1/opt/anaconda3/bin/activate
no change     /home/xihan1/opt/anaconda3/bin/deactivate
no change     /home/xihan1/opt/anaconda3/etc/profile.d/conda.sh
no change     /home/xihan1/opt/anaconda3/etc/fish/conf.d/conda.fish
no change     /home/xihan1/opt/anaconda3/shell/condabin/Conda.psm1
no change     /home/xihan1/opt/anaconda3/shell/condabin/conda-hook.ps1
no change     /home/xihan1/opt/anaconda3/lib/python3.12/site-packages/xontrib/conda.xsh
no change     /home/xihan1/opt/anaconda3/etc/profile.d/conda.csh
no change     /home/xihan1/.bashrc
No action taken.
Thank you for installing Anaconda3!

==> For changes to take effect, close and re-open your current shell. <==
```
- Conda initialization script from eager-load to lazy-load:
```bash
# >>> conda initialize (lazy-loaded for fast shell startup) >>>
# !! Contents within this block are managed by 'conda init' !!
__CONDA_ROOT="$HOME/opt/anaconda3"
conda() {
    unset -f conda
    __conda_setup="$("$__CONDA_ROOT/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$__CONDA_ROOT/etc/profile.d/conda.sh" ]; then
            . "$__CONDA_ROOT/etc/profile.d/conda.sh"
        else
            export PATH="$__CONDA_ROOT/bin:$PATH"
        fi
    fi
    unset __conda_setup
    conda "$@"
}
# <<< conda initialize <<<
```
- New virtual environment (do NOT use `opencv` package from `conda-forge`; go to PyPI):
```bash
# Stay with Python 3.10 for PyTorch 2.5.1 compatibility. 
conda create -n py3 python=3.10
conda activate py3

# Boilerplate packages. 
# for opencv-python's versioning, see below.
# for PyTorch's version under this default setting, see https://pytorch.org/get-started/locally/.
pip install matplotlib einops h5py numpy scipy sympy scikit-image scikit-learn opencv-python-headless

# Stay with the legacy versions of PyTorch via conda. 
# pip install torch torchvision torchaudio
# https://pytorch.org/get-started/locally/
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Ubuntu with AMD64 architecture is fine;
# MacOSX with ARM64 architecture does not have scikit-learn-intelex!
pip install scikit-learn-intelex

# Juypter.
pip install ipykernel 
python -m ipykernel install --name py3 --user
```
- **Never** update anaconda itself as well as the base environment, it is highly likely to downgrade!
```
# NEVER call the following commands:
conda update anaconda
conda update --all
```
- [Uninstall anaconda](https://docs.anaconda.com/anaconda/install/uninstall/)
```bash
conda install anaconda-clean
anaconda-clean --yes
# -rf needed; there are write-protected contents in ~/opt/anaconda3 and will pop up prompts. 
rm -rf ~/opt/anaconda3
# Remove Anaconda init script from .bash_rc
```
- `opencv-python`:
  - `opencv-python` conflicts with `matplotlib` (which uses `PyQt5` on its default `QtAgg1` backend) on Qt versioning. 
    - Could not load the Qt platform plugin "xcb". 
    - Check [this](https://github.com/opencv/opencv-python/issues/386#issuecomment-687655197) GitHub issue. 
      - Don't use PyQt and opencv-python together
      - Instead, use `opencv-python-headless` which does not have dependency to Qt.
    - Also check the following [opencv-python README](https://github.com/opencv/opencv-python). 
  - Ultimate solution:
    ```bash
    # pip uninstall opencv-python
    # pip install opencv-python-headless

    # A previous release of PyCharm fails to index `opencv-python>=4.6.0.66`:
    # Pycharm refused to index or autocomplete properly while on OpenCV version 4.6 (both main and contrib). 
    # Manually hint old versions when installing --> pip install opencv-python-headless==4.5.5.64
    ```
- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyGLM](https://pypi.org/project/PyGLM/)
```bash
pip install PyGLM
```
- **Note**: When building third-library C++ projects, remember to deactive the base conda environment to avoid dependency overlaps! 



### jupyter lab

#### generate config

```bash
jupyter server --generate-config
jupyter server password
```

edit `~/.jupyter/jupyter_server_config.py`:    

```
https://jupyter-notebook.readthedocs.io/en/stable/public_server.html?highlight=serverapp%20ip#running-a-public-notebook-server
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.ServerApp.ip = '*'                 # 357
c.ServerApp.open_browser = False     # 461
c.ServerApp.password = 'argon2:...'  # 465
c.ServerApp.port = 9000              # 473
```
    
#### kernel management

```
jupyter kernelspec list
jupyter kernelspec uninstall unwanted-kernel
```



## 🌱 C/C++ 

- **Note**: When building third-library C++ projects, remember to deactive the base conda environment to avoid dependency overlaps! 

### CLion

- `__CLION_IDE__`：在 CLion 的 CMakeLists.txt 以及程序中都可使用的宏
- `$ENV{USER}`, `$ENV{HOME}`： CMakeLists.txt 中调用系统变量
- [Data flow analysis timeout](https://youtrack.jetbrains.com/issue/CPP-17623): press shift in CLion quickly twice, then we have a search window, search "Registry..." and change the timeout key. 

### Latest `gcc` on `ubuntu 20.04 LTS`

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc g++ gcc-13 g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13
sudo update-alternatives --config gcc
```
Do **NOT** use these commands, as you would have to re-install all symbolic links manually:
```bash
# Do NOT use these commands!
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++
```

### Latest `CMake` on `ubuntu 20.04 LTS`

- [Kitware Repository](https://apt.kitware.com/): 
> This is Kitware, Inc.'s third-party APT repository, which we use for hosting our own Ubuntu packages, such as CMake.
>
> We currently support Ubuntu 20.04 and 22.04 on our repository. 
> The repositories support x86 (64-bit only) and ARM (32-bit and 64-bit).
> 
> To add the repository to your installation, run the kitware-archive.sh script, or do the following in order: 
```bash
# 2. Obtain a copy of our signing key:
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

# 3. Add the repository to your sources list and update.
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update

# 4. Install the kitware-archive-keyring package to ensure that your keyring stays up to date as we rotate our keys:
sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
sudo apt-get install kitware-archive-keyring

# 6. (Step 5 is optional and skipped; it's for registering the non-production release candidates)
sudo apt install cmake
```
- Do **NOT** use command `sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'`
  as it modifies `/etc/apt/sources.list`! 
  We want all kitware stuff in `/etc/apt/sources.list.d/`. 
- To UNDO: The repo could be corrupted and will fail the apt updates (invalid certificate). 
```bash
sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
sudo rm /usr/share/keyrings/kitware-archive-removed-keys.gpg 

sudo rm /etc/apt/sources.list.d/kitware.list
sudo rm /etc/apt/sources.list.d/kitware.list.distUpgrade
sudo rm /etc/apt/sources.list.d/kitware.list.save

sudo vi /etc/apt/sources.list
# Remove all kitware entries if present

sudo apt clean
sudo reboot

# Redo the installation steps
```

### `{fmt}`

- Download and compile the [`{fmt}` repository](https://fmt.dev/latest/index.html). 
- Some special tricks when [building the libary](https://fmt.dev/latest/usage.html). 
  - The default build generates position-dependent static libraries. 
  - Some clients (e.g. Python C/C++ extension) require 
    either dynamic libraries or position-independent static libraries. 
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ...
make -j32
```

### Libraries

```
# Eigen, OpenCV, TBB, Boost
sudo apt update
sudo apt install libeigen3-dev libopencv-dev libtbb-dev libboost-all-dev 

# gnuplot (Dependency of Matplot++)
sudo apt install gnuplot
```

### `CGAL`

- [Download CGAL for Linux](https://www.cgal.org/download/linux.html)
```bash
cd CGAL
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```
### `OpenGL`

- Installation
```
# OpenGL
sudo apt install libglm-dev libglew-dev libglfw3-dev mesa-utils libx11-dev libxi-dev libxrandr-dev 
```
- Check apt installed package version
```
apt policy <package>
```
- Check system `OpenGL`
```
sudo glxinfo | grep "OpenGL"
```
- Upgrade to OpenGL 4.1 on VMWare Workstation Pro 16.x Guest ubuntu OS: 
    - Requirements: `mesa >= 20.2` and Linux kernel `>= 5.8`
```
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt-get dist-upgrade
sudo apt autoremove
sudo reboot
```

### [NVIDIA AMGX](https://github.com/NVIDIA/AMGX)

```bash
git clone --recursive git@github.com:nvidia/amgx.git
cd amgx
mkdir build
cd build
cmake .. -DCUDA_ARCH="70" -DCMAKE_NO_MPI=1 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_COMPILER="gcc" -DCMAKE_CXX_COMPILER="g++"
make -j32
```
```cmake
set(AMGX_INCLUDE_DIRS "$ENV{HOME}/lib/amgx/include/")
set(AMGX_LIBRARIES "$ENV{HOME}/lib/amgx/build/libamgx.a")
```

### [Matplot++](https://github.com/alandefreitas/matplotplusplus)

- [Install Binary Packages](https://alandefreitas.github.io/matplotplusplus/integration/install/binary-packages/)
  - Releases -> `matplotplusplus-1.1.0-Linux.tar.gz`. 
  - Do **not** compile from source; `std::filesystem` not found error for whatever reason.
- Integration with CMake:
```cmake
find_package(Matplot++ REQUIRED HINTS "$ENV{HOME}/lib/matplotplusplus/lib/cmake/Matplot++/")
target_link_libraries(${TARGET} Matplot++::matplot)
```
- Do **NOT** Use [This One](https://alandefreitas.github.io/matplotplusplus/integration/cmake/install-as-a-package-via-cmake/)!
```bash
# DO NOT DO THIS!

sudo apt install gnuplot

git clone https://github.com/alandefreitas/matplotplusplus.git

cmake -B build/local \
    -DMATPLOTPP_BUILD_EXAMPLES=OFF \
    -DMATPLOTPP_BUILD_SHARED_LIBS=ON \
    -DMATPLOTPP_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$HOME/lib/Matplot++" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

cmake --build build/local; cmake --install build/local

# Note: 
# - `cmake --build build/local` automatically installs `libmatplot`. 
# - The actual install path is: 
#   - `${CMAKE_INSTALL_PREFIX}/include`;
#   - `${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}` where `CMAKE_INSTALL_LIBDIR=lib`. 
```

### `OpenMesh`

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/ax/lib/OpenMesh
make -j4
make install
```

### `Qt`

- Installation
```
# Qt
sudo apt install qt5-default
sudo apt install qtcreator
```
- `CMake` Options: 
```
# These 3 options for Qt support
# You need to add your header files in add_executable, otherwise the moc won't parse them
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ...

# Disable Qt's bad marco usage to avoid conflicts! 
# After this, call original keywords such as: slots -> Q_SLOTS
set(ALL_COMPILE_DEFS
        -DQT_NO_KEYWORDS
        )
```
- DO **include headers** in `add_executable` command, 
  or `moc` will NOT parse them and there will be problems finding `vtable`!

### [pybind11](https://github.com/pybind/pybind11)

- This is used to build C/C++ source code into a shared library (.so) which could be imported in Python. 
- When using with CMake, simply use the `pybind11_add_module` command to replace CMake's `add_library` command. 
- No need to `add_executable`. No program runnable at C/C++ side. 
- Dynamic or Position-independent static library required at C/C++ side. 
  - See `{fmt}`'s [build guide](https://fmt.dev/latest/usage.html) as an example. 
  - Details also available above in `{fmt}`'s section. 
- See the [Sample CMakeLists.txt](../../code/CPX/CMakeLists.txt). 

## 🌱 Java

### jdk

```
tar -zxvf jdk-11.0.7_linux-x64_bin.tar.gz
sudo mv jdk-11.0.7 /opt/jdk

sudoedit /etc/profile.d/my-env-vars.sh

# jvm
export JAVA_HOME="/opt/jdk"
export JRE_HOME="${JAVA_HOME}/jre"
export CLASSPATH=".:${JAVA_HOME}/lib:${JRE_HOME}/lib:${CLASSPATH}"
export PATH="${JAVA_HOME}/bin:${JRE_HOME}/bin${PATH:+:${PATH}}"
```

### jdk的安装位置

From Baidu: 

> 如果你认为`jdk`是系统提供给你可选的程序，放在`/opt`里；
> 如果你认为这是你个人行为，自主安装的，放在`/usr/local/lib`里；
> 如果你觉得`jdk`对你来说是必不可少的运行库，放在`/lib`里。

> 上面三句是最开始的想法。

> 其实我也想找出一个最佳实践，后来看了看`linux`的目录结构，发现：就算是同一个东西，系统自带和你手动安装，就不应该在同一个目录里。
> 同样是浏览器，系统自带的`firefox`就在`/usr/lib`里，而后来通过软件包安装的`chrome`就在`/opt`里。
> 如果系统自带`java`，我觉得他会在`/lib`里或者`/usr/lib`，看它对`java`的定义是不是必需的库。
> 如果能在软件管理器里安装，那么会安装在`/usr/lib`；
> 如果`oracle`给你的是`deb`，那么会被安装在`opt`；
> 所以自己安装，就要放在`/usr/local/lib`里比较合适了。

### IntelliJ IDEA

#### hotkeys

- 提示：基本提示Ctrl+Space，按类型信息提示Ctrl+Shift+Space
- 修复：快速修复Alt+Enter，补全末尾字符Ctrl+Shift+Enter
- 重构：重构汇总Ctrl+Shift+Alt+T，变量改名Shift+F6，提取变量Ctrl+Alt+V
- 格式化：格式化import列表Ctrl+Alt+O，格式化代码Ctrl+Alt+L
- 运行：运行程序Alt+Shift+F10，启动调试Shift+F9
- 方法分隔符：【File】→【Settings...】→【Editor】→【General】→【Appearance】→【Show method separators】
- 模板提示：`Ctrl + J`即可选择插入根据上下文预定义好的代码片段
- Surround With: in the Code menu, click Surround With `Ctrl + Alt + T`.
- Press `Ctrl + \`` to execute the View | Quick Switch Scheme action.

#### heap size

```
gedit -idea.vmoptions
gedit -idea64.vmoptions
...
-Xms2048m
-Xmx8192m
```



## 🌱 Javascript

### `Node.js`

`npm` comes with `node.js`: [NodeSource Node.js Binary Distributions](https://github.com/nodesource/distributions/blob/master/README.md)

### `yarn`

[Installation | Yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)



## 🌱 `MongoDB`

[Installation | MongoDB Community Edition](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
