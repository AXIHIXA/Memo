# Memo

## new ubuntu

### dependencies

    sudo apt install vim
    sudo apt install tmux

    touch ~/.tmux.conf
    set -g mouse on
    set -g status-interval 60
    set -g display-time 3000
    set -g history-limit 65535

## python

### anaconda

    conda create -n py3 python=3.7 numpy scipy sympy matplotlib cython ipykernel h5py
    source activate py3
    pip install opencv-python
    python -m ipykernel install --name py3 --user
    
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

    tar -zxvf jdk-8u201-linux-x64.tar.gz
    sudo mv jdk1.8.0_201 /usr/local/lib/jdk

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

#### training
    
    sudo gedit /etc/hosts
    # JetBrains
    0.0.0.0 account.jetbrains.com
    
    gedit -idea.vmoptions
    gedit -idea64.vmoptions
    -javaagent:/home/ax/opt/idea/lib/JetbrainsIdesCrack-4.2.jar
    ...
    -Xms1024m
    -Xmx2048m
    -XX:ReservedCodeCacheSize=1024m
