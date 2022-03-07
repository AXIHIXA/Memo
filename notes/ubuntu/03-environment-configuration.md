# Environment Configuration


## 🌱 Terminal 

### Shortcuts

- `Ctrl` + `u`: Delete all text from cursor to head
- `Ctrl` + `k`: Delete all text from cursor to tail
- `Ctrl` + `w`: Delete one word ahead of cursor
- `Ctrl` + `a`: Move cursor to head
- `Ctrl` + `e`: Move cursor to end
- `Alt` + `f`: Move cursor forward (towards head) by one word
- `Alt` + `b`: Move cursor back (towards tail) by one word

### Useful Commands

- `man -k "copy files`: see man page of all commands with "copy files"
- `!`
- `df -h`: See storage occupation of all mounted dirs; where `-h` convert unit into human-readable (MB, GB...)
- `du -h --max-depth=1 /home` (or `-d 1`): See storage occupation of `/home`
- `free -h`: See current RAM usage
- `pgrep hello`, `pidof hello`: find pid of process "hello"
- `killall hello`, `pkill hello`: kill all processes called "hello"

## System Utils

### `apt` PPA Management

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

- Installation:
```
# `bash` is CRITICAL!!! or you may fail when choosing to modify .bashrc
bash ./Anaconda...
```
- Choose `yes`:
```
Do you wish the installer to initialize Anaconda3
by running conda init? [yes|no]
[no] >>> yes
no change     /home/ax/opt/anaconda3/condabin/conda
no change     /home/ax/opt/anaconda3/bin/conda
no change     /home/ax/opt/anaconda3/bin/conda-env
no change     /home/ax/opt/anaconda3/bin/activate
no change     /home/ax/opt/anaconda3/bin/deactivate
no change     /home/ax/opt/anaconda3/etc/profile.d/conda.sh
no change     /home/ax/opt/anaconda3/etc/fish/conf.d/conda.fish
no change     /home/ax/opt/anaconda3/shell/condabin/Conda.psm1
no change     /home/ax/opt/anaconda3/shell/condabin/conda-hook.ps1
no change     /home/ax/opt/anaconda3/lib/python3.7/site-packages/xontrib/conda.xsh
no change     /home/ax/opt/anaconda3/etc/profile.d/conda.csh
modified      /home/ax/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false

Thank you for installing Anaconda3!

===========================================================================

Anaconda and JetBrains are working together to bring you Anaconda-powered
environments tightly integrated in the PyCharm IDE.

PyCharm for Anaconda is available at:
https://www.anaconda.com/pycharm
```
- New virtual environment (do NOT use `opencv` package from `conda`. It's bullsh*t):
```
conda create -n py3 python=3.8 numpy scipy sympy matplotlib cython ipykernel
conda activate py3
pip install opencv-python
python -m ipykernel install --name py3 --user
```
- **Never** update anaconda itself as well as the base environment, it is highly likely to downgrade!
    - **Never** call the following commands:
    ```
    conda update anaconda
    conda update --all
    ```
- [Uninstall anaconda](https://docs.anaconda.com/anaconda/install/uninstall/)
```
conda install anaconda-clean
anaconda-clean --yes
rm -rf ~/opt/anaconda3
# Remove Anaconda path from .bash_profile
```
    
#### `.bashrc`

##### `VMWare` Client

```
gedit ~/.bashrc

# anaconda
alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd /media/ax/DATA/workspace"
alias jl="cd /media/ax/DATA/workspace; jupyter lab"
alias cls="reset"
```

##### `WSL`

```
gedit ~/.bashrc

# anaconda
alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd /mnt/d/workspace"
alias jl="cd /mnt/d/workspace; jupyter lab"
alias ss="sudo service ssh --full-restart"
alias cls="reset"
```
    
### jupyter lab

#### generate config

```
jupyter server --generate-config
jupyter server password
```

edit `jupyter_server_config.py`:    

```
c.ServerApp.ip = '0.0.0.0'         # 278
c.ServerApp.open_browser = False   # 366
c.ServerApp.password = 'sha1:...'  # 377
c.ServerApp.port = 9000            # 390
```
    
#### kernel management

```
jupyter kernelspec list
jupyter kernelspec uninstall unwanted-kernel
```

## 🌱 java

### jdk

```
tar -zxvf jdk-11.0.7_linux-x64_bin.tar.gz
sudo mv jdk-11.0.7 /opt/jdk

sudo gedit /etc/profile

# jvm
export JAVA_HOME=/opt/jdk
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib:${CLASSPATH}
export PATH=${JAVA_HOME}/bin:${JRE_HOME}/bin:${PATH}
```

### jdk的安装位置

引用自百度知道（这地方整体不咋地，不过这句话我感觉还是挺靠谱的）：

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

## 🌱 C/C++ 

### CLion

- `__CLION_IDE__`：在 CLion 的 CMakeLists.txt 以及程序中都可使用的宏
- `$ENV{USER}`： CMakeLists.txt 中调用系统变量
- [Data flow analysis timeout](https://youtrack.jetbrains.com/issue/CPP-17623): press shift in CLion quickly twice, then we have a search window, search "Registry..." and change the timeout key. 

### `gcc-9` on `ubuntu 18.04 LTS`
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9 g++-9
```

### `gcc-11-multilib` on `ubuntu 20.04 LTS`

Follow the instructions from 
[here](https://packages.ubuntu.com/hirsute/amd64/gcc-11-multilib/download), 
which is to:
- Update the listed mirrors by adding a line to your `/etc/apt/sources.list` like:
- `sudo add-apt-repository 'deb http://mirrors.kernel.org/ubuntu hirsute main universe'`
- Choose a mirror based on your location from the list. I chose the kernel mirror as I am in North America.
- `sudo apt-get update`
- `sudo apt-get install gcc-11`
After that which `gcc-11` should produce a path to `gcc-11`. 
```bash
$ which gcc-11
/usr/bin/gcc-11
```

```bash
$ sudo update-alternatives --config gcc
There are 6 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path             Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-11   1010      auto mode
  1            /usr/bin/gcc-10   1000      manual mode
  2            /usr/bin/gcc-11   1010      manual mode
  3            /usr/bin/gcc-5    40        manual mode
  4            /usr/bin/gcc-7    700       manual mode
  5            /usr/bin/gcc-8    800       manual mode
  6            /usr/bin/gcc-9    900       manual mode

Press <enter> to keep the current choice[*], or type selection number:
```

### Libraries

```
# Eigen, OpenCV, TBB, Boost
sudo apt update
sudo apt install libeigen3-dev libopencv-dev libtbb-dev libboost-all-dev 

# CGAL
sudo apt install libcgal-dev libcgal-qt5-dev
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

### `CUDA`

- Install NVIDIA `CUDA` Toolkit: 
```
sudo apt install nvidia-cuda-toolkit
```
- Install `CUDNN`: Follow instructions on 
[NVIDIA CUDNN DOCUMENTAZTION](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb)
(I am using Debian Installation. Do NOT use Package Manager Installation. )

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
- DO **include headers** in `add_executable` command, or `moc` will NOT parse them and there will be problems finding `vtable`!


## 🌱 Javascript

### `Node.js`

`npm` comes with `node.js`: [NodeSource Node.js Binary Distributions](https://github.com/nodesource/distributions/blob/master/README.md)

### `yarn`

[Installation | Yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)


## 🌱 `MongoDB`

[Installation | MongoDB Community Edition](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
