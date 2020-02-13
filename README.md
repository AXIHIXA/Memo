# Memo

## new ubuntu

### driver problem

Taken from https://www.cnblogs.com/deepllz/p/8892628.html as personal note. 
    
#### 从U盘启动安装ubuntu
- 首先进入BIOS（dell笔记本开机按F2，其他型号电脑请百度）关闭win10的secure boot，然后在电源选项中禁用快速启动（win10的快速启动在电源选项中，具体路径为：电源选项→选择电源按钮的功能→更改当前不可用设置→去掉启用快速启动前面的√，保存即可）
- 然后再进入boot options（DELL笔记本开机按F12），选择从U盘启动
- 在选择install ubuntu之前，按e进入grub编辑页面，将倒数第二行的 quiet splash 替换成 nomodeset（后面安装成功，并利用软件更新器更新了软件后，启动黑屏问题也是这样解决），按F10保存，即可成功进入安装程序
- 安装过程中的磁盘分区（其他步骤直接点继续就可以了，所以说一下如何进行磁盘分区）在进入到安装类型（install type）这一步骤的时候，建议选择其他（something else）手动进行分区，这样可以自己控制磁盘的分配。手动分区要在空闲分区（free space）分配，点击"+"号即可分配，分配的大小类型如下所示:　 
        
        
        partition    file system    size
        efi          --             300M
        swap         swap           32G
        /boot        ext4           1G
        /            ext4           remaining


- 分区之后要将boot所对应的分区设置为启动引导器，否则启动时操作系统可能找不到引导项导致无法进入ubuntu
　　 
- The grub page is default hidden if only ubuntu is detected. To see this page and edit grub after installation, boot with efi floppy inserted. 
　　 
- update nvidia drivers: 
        
        
        $ ubuntu-drivers devices
        == /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
        modalias : pci:v000010DEd00001180sv00001458sd0000353Cbc03sc00i00
        vendor   : NVIDIA Corporation
        model    : GP106 [GeForce GTX 1060 6GB]
        driver   : nvidia-304 - distro non-free
        driver   : nvidia-340 - distro non-free
        driver   : nvidia-390 - distro non-free recommended
        driver   : xserver-xorg-video-nouveau - distro free builtin

        == cpu-microcode.py ==
        driver   : intel-microcode - distro free
        　　 
        $ sudo ubuntu-drivers autoinstall


#### 配置ssh

	
	sudo apt install openssl-server
	dpkg -l | grep ssh  # 应该看到 openssh-server
	ps -e | grep ssh  # 应该看到 sshd
	

- 如果看到sshd那说明ssh-server已经启动了。
- 如果没有则可以这样启动：

```
sudo /etc/init.d/ssh stop
sudo /etc/init.d/ssh start
```

- 配置相关：
ssh-server配置文件位于/etc/ssh/sshd_config，在这里可以定义SSH的服务端口，默认端口是22，你可以自己定义成其他端口号，如222。
（或把配置文件中的”PermitRootLogin without-password”加一个”#”号，把它注释掉，再增加一句”PermitRootLogin yes”），然后重启SSH服务：

```
sudo /etc/init.d/ssh stop
sudo /etc/init.d/ssh start
```


- 此时已经可以`ssh`登录同时支持`sftp`。

- ssh免密码登录：将需要免密码的机器的ssh公钥`id_rsa.pub` mv 至`/home/<user>/.ssh/authorized_ssh`。


#### 新建用户

	
	sudo adduser <user>  # 按照提示来，一般默认就行
	

- 查看用户：`cat /etc/passwd`
- 删除用户：`deluser`
- 设置`root`密码：`sudo passwd root`
- 添加`sudoers`：

```
sudo su - 
visudo

# Allow members of group sudo to execute any command
%sudo   ALL=(ALL:ALL) ALL
ax      ALL=(ALL:ALL) ALL
```

- 添加免密`sudoers`：

```
sudo su - 
visudo

# Allow members of group sudo to execute any command
%sudo   ALL=(ALL:ALL) ALL
ax      ALL=(ALL:ALL) NOPASSWD : ALL
```

- `su`  后面不加用户是默认切到 `root`
- `su`  是不改变当前变量（only `.bashrc` will be sourced）
- `su -` 是改变为切换到用户的变量（expericene a login process, usually `.bash_profile` and `.bashrc` will be sourced）
- 也就是说`su`只能获得`root`的执行权限，不能获得环境变量；而`su -`是切换到`root`并获得`root`的环境变量及执行权限

#### 默认编辑器

```
vi etc/profile
# add following line 
export EDITOR="/usr/bin/vim"
```

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
    # Keep same as real ubuntu system mount 
    .host:/    /media/ax/DATAIN    fuse.vmhgfs-fuse    defaults,allow_other,uid=1000     0    0

### Application Entry

    # global
    sudo touch /home/ax/.local/share/applications/understand.desktop
    
    # current user
    touch /home/ax/.local/share/applications/understand.desktop
    gedit /home/ax/.local/share/applications/understand.desktop
    
    [Desktop Entry]
    Version=1.0
    Type=Application
    Name=understand
    Icon=/home/ax/opt/understand/bin/linux64/understand_64.png
    Exec="/home/ax/opt/understand/bin/linux64/understand"
    Categories=Development;IDE;
    Terminal=false

### ex****** passport

***For people getting the white screen error, go to Library > History > Show All History, right click on ex******, and Forget About This Site***

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
    -Xms2048m
    -Xmx8192m

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
    
### tbb

    sudo apt install libtbb-dev
    
### boost + cgal

    I suggest not using this holy s**t thing. 

    sudo apt install libboost-all-dev libcgal-dev libcgal-qt5-dev
