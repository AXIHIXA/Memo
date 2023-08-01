# Environment Configuration



## 🌱 Environment Variables

### Wiki

- See [Environment Variables - Ubuntu Documentation](https://help.ubuntu.com/community/EnvironmentVariables). 
- Local Variables: 
  - Suggested: `~/.profile`. 
    - **Not** for anaconda initialization script and aliases! 
      These still go to `~/.bashrc`. 
  - Shell config files such as `~/.bashrc`, `~/.bash_profile`, and `~/.bash_login` 
    are often suggested for setting environment variables. 
    While this may work on Bash shells for programs started from the shell, 
    variables set in those files are **not** available 
    by default to programs started from the graphical environment in a desktop session.
- System Variables: 
  - Suggested: `/etc/profile.d/*.sh`. 
  - While `/etc/profile` is often suggested for setting environment variables system-wide, 
    it is a configuration file of the base-files package, 
    so it's **not** appropriate to edit that file directly. 
    Use a file in `/etc/profile.d` instead as shown above. 
    (Files in `/etc/profile.d` are sourced by `/etc/profile`.)
  - The shell config file `/etc/bash.bashrc` is sometimes suggested for setting environment variables system-wide. 
    While this may work on Bash shells for programs started from the shell, 
    variables set in that file are **not** available 
    by default to programs started from the graphical environment in a desktop session.
- `sudo` Caveat:
  - Any variables added to these locations will **not** be reflected when invoking them with a `sudo` command, 
    as `sudo` has a default policy of resetting the Environment 
    and setting a secure path (this behavior is defined in `/etc/sudoers`). 
    - As a workaround, you can use `sudo su` that will provide a shell with root privileges 
      but retaining any modified `PATH` variables. 
    - Alternatively you can setup sudo not to reset certain environment variables 
      by adding some explicit environment settings to keep in `/etc/sudoers`. 

### Current `/etc/profile.d/my-env-vars.sh`

```bash
sudo touch /etc/profile.d/my-env-vars.sh
sudoedit /etc/profile.d/my-env-vars.sh

# /etc/profiles.d/my-env-vars.sh

# cuda
export PATH="/usr/local/cuda/bin:${PATH}"

# editor
export EDITOR="/usr/bin/vim"

# jvm
export JAVA_HOME="/opt/jdk"
export JRE_HOME="${JAVA_HOME}/jre"
export CLASSPATH=".:${JAVA_HOME}/lib:${JRE_HOME}/lib:${CLASSPATH}"
export PATH="${JAVA_HOME}/bin:${JRE_HOME}/bin:${PATH}"
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
- New virtual environment (do NOT use `opencv` package from `conda-forge`; go to PyPI):
```
conda create -n py3 ipykernel matplotlib numpy scipy scikit-image scikit-learn scikit-learn-intelex sympy 
conda activate py3
python -m ipykernel install --name py3 --user
# for opencv-python, see below. 
```
- **Never** update anaconda itself as well as the base environment, it is highly likely to downgrade!
    - **Never** call the following commands:
    ```
    conda update anaconda
    conda update --all
    ```
- [Uninstall anaconda](https://docs.anaconda.com/anaconda/install/uninstall/)
```bash
conda install anaconda-clean
anaconda-clean --yes
rm -rf ~/opt/anaconda3
# Remove Anaconda path from .bash_rc
```
- `opencv-python`:
  - PyCharm fails to index `opencv-python>=4.6.0.66`:
    - Pycharm refused to index or autocomplete properly while on OpenCV version 4.6 (both main and contrib). 
    - Manually hint old versions when installing. 
  - `opencv-python` conflicts with `matplotlib` (which uses `PyQt5` on its default `QtAgg1` backend) on Qt versioning. 
    - Could not load the Qt platform plugin "xcb". 
    - Check [this](https://github.com/opencv/opencv-python/issues/386#issuecomment-687655197) GitHub issue. 
      - Don't use PyQt and opencv-python together
      - Instead, use `opencv-python-headless` which does not have dependency to Qt.
    - Also check the following [opencv-python README](https://github.com/opencv/opencv-python). 
  - Ultimate solution:
    ```bash
    # pip uninstall opencv-python
    pip install opencv-python-headless==4.5.5.64
    ```
- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyGLM](https://pypi.org/project/PyGLM/)
```bash
pip install PyGLM
```



#### `.bashrc`

##### Real Ubuntu System

```bash
# anaconda
alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd /home/xihan1/workspace"
alias jl="cd /home/xihan1/workspace; jupyter lab"
alias cls="reset"
```

##### `VMWare` Client

```bash
gedit ~/.bashrc

# anaconda
alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd /media/ax/DATA/workspace"
alias jl="cd /media/ax/DATA/workspace; jupyter lab"
alias cls="reset"
```

##### `WSL`

```bash
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

```bash
jupyter server --generate-config
jupyter server password
```

edit `jupyter_server_config.py`:    

```
https://jupyter-notebook.readthedocs.io/en/stable/public_server.html?highlight=serverapp%20ip#running-a-public-notebook-server
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.ServerApp.ip = '*'                 # 282
c.ServerApp.open_browser = False     # 383
c.ServerApp.password = 'argon2:...'  # 394
c.ServerApp.port = 9000              # 407
```
    
#### kernel management

```
jupyter kernelspec list
jupyter kernelspec uninstall unwanted-kernel
```



## 🌱 C/C++ 

### CLion

- `__CLION_IDE__`：在 CLion 的 CMakeLists.txt 以及程序中都可使用的宏
- `$ENV{USER}`, `$ENV{HOME}`： CMakeLists.txt 中调用系统变量
- [Data flow analysis timeout](https://youtrack.jetbrains.com/issue/CPP-17623): press shift in CLion quickly twice, then we have a search window, search "Registry..." and change the timeout key. 

### `gcc-13` on `ubuntu 20.04 LTS`

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc g++ gcc-13 g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13
sudo update-alternatives --config gcc
```
Do not use these commands as you would have to install all symbolic links again:
```bash
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++
```

### Latest `CMake` on `ubuntu 20.04 LTS`

- [Kitware Repository](https://apt.kitware.com/): 
> This is Kitware, Inc.'s third-party APT repository, which we use for hosting our own Ubuntu packages, such as CMake.
>
> We currently support Ubuntu 16.04, 18.04, and 20.04 on our repository. 
> The 16.04 and 18.04 repositories support x86 (32-bit and 64-bit), 
> and the 20.04 repository supports x86 (32-bit and 64-bit) and ARM (32-bit and 64-bit).
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

# 6. 
sudo apt install cmake
```

### `{fmt}`

Download and compile the [`{fmt}` repository](https://fmt.dev/latest/index.html). 

### [Matplot++](https://github.com/alandefreitas/matplotplusplus)

- [Install Binary Packages](https://alandefreitas.github.io/matplotplusplus/integration/install/binary-packages/)
  - Releases -> `matplotplusplus-1.1.0-Linux.tar.gz`. 
  - Do **not** compile from source; `std::filesystem` not found error for whatever reason.
- Integration with CMake:
```cmake
find_package(Matplot++ REQUIRED HINTS "$ENV{HOME}/lib/matplotplusplus/lib/cmake/Matplot++/")
target_link_libraries(${TARGET} Matplot++::matplot)
```

### Libraries

```
# Eigen, OpenCV, TBB, Boost
sudo apt update
sudo apt install libeigen3-dev libopencv-dev libtbb-dev libboost-all-dev 

# gnuplot (Dependency of Matplot++)
sudo apt install gnuplot

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

- Install Nvidia driver: Refer to [01-system-installation.md](./01-system-installation.md)
- Install [NVIDIA `CUDA` Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local): 
```
# DO NOT USE THIS OBSELETE VERSION!
sudo apt install nvidia-cuda-toolkit
```
- Set environment variables: 
```bash
sudoedit /etc/profile.d/my-env-vars.sh

# cuda
export PATH="/usr/local/cuda/bin:${PATH}"
```
- Install `CUDNN`: Follow instructions on 
[NVIDIA CUDNN DOCUMENTAZTION](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb)
(I am using Debian Installation. Do NOT use Package Manager Installation. )
- CudaDemo: [CudaDemo](../../code/CudaDemo)
- To uninstall, follow the [offical guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-tk-and-driver), which ensures that the uninstalltion will be clean. 

### [MatPlot++](https://alandefreitas.github.io/matplotplusplus/integration/cmake/install-as-a-package-via-cmake/)

```bash
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
```
Note: 
- `cmake --build build/local` automatically installs `libmatplot`. 
- The actual install path is: 
  - `${CMAKE_INSTALL_PREFIX}/include`;
  - `${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}` where `CMAKE_INSTALL_LIBDIR=lib`. 

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



## 🌱 java

### jdk

```
tar -zxvf jdk-11.0.7_linux-x64_bin.tar.gz
sudo mv jdk-11.0.7 /opt/jdk

sudoedit /etc/profile.d/my-env-vars.sh

# jvm
export JAVA_HOME="/opt/jdk"
export JRE_HOME="${JAVA_HOME}/jre"
export CLASSPATH=".:${JAVA_HOME}/lib:${JRE_HOME}/lib:${CLASSPATH}"
export PATH="${JAVA_HOME}/bin:${JRE_HOME}/bin:${PATH}"
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



## 🌱 Javascript

### `Node.js`

`npm` comes with `node.js`: [NodeSource Node.js Binary Distributions](https://github.com/nodesource/distributions/blob/master/README.md)

### `yarn`

[Installation | Yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)



## 🌱 `MongoDB`

[Installation | MongoDB Community Edition](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
