# Environment Configuration

## python

### anaconda

```
conda create -n py3 python=3.7 numpy scipy sympy matplotlib cython ipykernel h5py
conda activate py3
pip install opencv-python PrettyTable
python -m ipykernel install --name py3 --user
```
    
#### `.bashrc`

##### `VMWare` Client

```
gedit ~/.bashrc

# anaconda
conda deactivate
export PATH="/home/ax/opt/anaconda3/bin:${PATH}"

alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd /media/ax/DATAIN/workspace"
alias jl="cd /media/ax/DATAIN/workspace; jupyter lab"
```

##### `WSL`

```
gedit ~/.bashrc

# anaconda
conda deactivate
export PATH="/home/ax/opt/anaconda3/bin:${PATH}"

alias c3="conda activate py3"
alias d3="conda deactivate"
alias ws="cd /mnt/d/workspace"
alias jl="cd /mnt/d/workspace; jupyter lab"
```
    
### jupyter lab

#### generate config

```
jupyter notebook --generate-config
jupyter notebook password
```

edit `jupyter_notebook_config.py`:    

```
c.NotebookApp.ip = '0.0.0.0'         # 204
c.NotebookApp.open_browser = False   # 272
c.NotebookApp.password = 'sha1:...'  # 281
c.NotebookApp.port = 9000            # 292
```
    
#### view installed kernels

```
jupyter kernelspec list
```

## java

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

#### heap size

```
gedit -idea.vmoptions
gedit -idea64.vmoptions
...
-Xms2048m
-Xmx8192m
```

## C/C++ 

### Libraries

```
# Eigen
sudo apt install libeigen3-dev

# OpenCV
sudo apt install libopencv-dev

# OpenGL
sudo apt install freeglut3-dev libglm-dev libglew-dev libglfw3-dev libxmu-dev libxi-dev

# Qt
sudo apt install qt5-default
sudo apt install qtcreator

# tbb
sudo apt install libtbb-dev

# boost + cgal
# I suggest not using this holy s\*\*t thing. 
sudo apt install libboost-all-dev libcgal-dev libcgal-qt5-dev
```

## Javascript

### `Node.js`

`npm` comes with `node.js`: [NodeSource Node.js Binary Distributions](https://github.com/nodesource/distributions/blob/master/README.md)

### `yarn`

[Installation | Yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)

## Some Alienware Key Shortcut

```
Fn + F6 : disables / enables WIN key
```