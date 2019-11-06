# Memo

## new ubuntu

### driver problem

    https://www.cnblogs.com/deepllz/p/8892628.html

### time fix

    sudo timedatectl set-local-rtc 1 --adjust-system-clock

### dependencies

    sudo apt install vim
    sudo apt install tmux

    touch ~/.tmux.conf
    set -g mouse on
    set -g status-interval 60
    set -g display-time 3000
    set -g history-limit 65535
    
### VMWare Shared Folder

    # https://askubuntu.com/questions/29284/how-do-i-mount-shared-folders-in-ubuntu-using-vmware-tools 
    
    sudo gedit /etc/fstab
    # Use shared folders between VMWare guest and host
    .host:/    /mnt/hgfs/    fuse.vmhgfs-fuse    defaults,allow_other,uid=1000     0    0

## python

### anaconda

    conda create -n py3 python=3.7 numpy scipy sympy matplotlib cython ipykernel h5py
    source activate py3
    pip install opencv-python
    python -m ipykernel install --name py3 --user
    
#### Note

    If you'd prefer that conda's base environment not be activated on startup,
       set the auto_activate_base parameter to false:

    conda config --set auto_activate_base false

    Thank you for installing Anaconda3!
    
#### .bashrc

    # conda deactivate
    conda deactivate
    export PATH="/home/ax/opt/anaconda3/bin:$PATH"
    
### jupyter lab

#### generate config

    jupyter notebook --generate-config
    jupyter notebook password
    
edit `jupyter_notebook_config.py`:    
    
    c.NotebookApp.ip = '0.0.0.0'         # 204
    c.NotebookApp.open_browser = False   # 267
    c.NotebookApp.password = 'sha1:...'  # 276
    c.NotebookApp.port = 9000            # 287
    
#### view installed kernels

    jupyter kernelspec list

## java

### jdk

    tar -zxvf jdk-11.0.5_linux-x64_bin.tar.gz
    sudo mv jdk-11.0.5 /usr/local/lib/jdk

    sudo gedit /etc/profile
    # jvm
    export JAVA_HOME=/usr/local/lib/jdk
    export JRE_HOME=${JAVA_HOME}/jre
    export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib:${CLASSPATH}
    export PATH=${JAVA_HOME}/bin:${JRE_HOME}/bin:${PATH}

### IntelliJ IDEA

#### hotkeys

- 提示：基本提示Ctrl+Space，按类型信息提示Ctrl+Shift+Space
- 修复：快速修复Alt+Enter，补全末尾字符Ctrl+Shift+Enter
- 重构：重构汇总Ctrl+Shift+Alt+T，变量改名Shift+F6，提取变量Ctrl+Alt+V
- 格式化：格式化import列表Ctrl+Alt+O，格式化代码Ctrl+Alt+L
- 运行：运行程序Alt+Shift+F10，启动调试Shift+F9
- 方法分隔符：【File】→【Settings...】→【Editor】→【General】→【Appearance】→【Show method separators】

#### heap size
   
    gedit -idea.vmoptions
    gedit -idea64.vmoptions
    ...
    -Xms1024m
    -Xmx2048m

## C/C++

### Eigen

    sudo apt install libeigen3-dev

### OpenCV

    sudo apt install libopencv-dev

### OpenGL
    
    sudo apt install freeglut3-dev libglm-dev libglew-dev libglfw3-dev libxmu-dev libxi-dev
    
### Qt
    
    sudo apt install qt5-default
    sudo apt install qtcreator
    
### SFML

    sudo apt install libsfml-dev
    
### boost + cgal

    I suggest not using this holy s**t thing. 

    sudo apt install libboost-all-dev libcgal-dev libcgal-qt5-dev
