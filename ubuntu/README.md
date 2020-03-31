# new ubuntu

## installation

Taken from https://www.cnblogs.com/deepllz/p/8892628.html as personal note. 

- 首先进入BIOS（dell笔记本开机按F2，其他型号电脑请百度）关闭win10的secure boot，然后在电源选项中禁用快速启动（win10的快速启动在电源选项中，具体路径为：电源选项→选择电源按钮的功能→更改当前不可用设置→去掉启用快速启动前面的√，保存即可）
- 然后再进入boot options（DELL笔记本开机按F12），选择从U盘启动
- 在选择install ubuntu之前，按e进入grub编辑页面，将倒数第二行的 quiet splash 替换成 nomodeset（后面安装成功，并利用软件更新器更新了软件后，启动黑屏问题也是这样解决），按F10保存，即可成功进入安装程序
- 安装过程中的磁盘分区（其他步骤直接点继续就可以了，所以说一下如何进行磁盘分区）在进入到安装类型（install type）这一步骤的时候，建议选择其他（something else）手动进行分区，这样可以自己控制磁盘的分配。手动分区要在空闲分区（free space）分配，点击"+"号即可分配，分配的大小类型如下所示:　 
        
```
partition    file system    size
efi          --             300M
swap         swap           32G
/boot        ext4           1G
/            ext4           remaining
```

- 分区之后要将boot所对应的分区设置为启动引导器，否则启动时操作系统可能找不到引导项导致无法进入ubuntu
　　 
- The grub page is default hidden if only ubuntu is detected. To see this page and edit grub after installation, boot with efi floppy inserted. 
　　 
- update nvidia drivers: 
        
```
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
```

## new system configuration


### WSL `.bashrc`

```
-rw-r--r--  1 ax   ax   3771 Mar 31 13:58 .bashrc
```

### dependencies

```
sudo apt install vim tmux git gcc g++ cmake gdb

touch ~/.tmux.conf
gedit ~/.tmux.conf

set -g mouse on
set -g status-interval 60
set -g display-time 3000
set -g history-limit 65535

ssh-keygen
git config --global user.name "Xi Han"
git config --global user.email "ahanxi@126.com"

sudo gedit /etc/profile

# default editor
export EDITOR="/usr/bin/vim"
```

### time fix

```
sudo timedatectl set-local-rtc 1 --adjust-system-clock
```
    
### VMWare Shared Folder

```
sudo gedit /etc/fstab
# Use shared folders between VMWare guest and host
# Keep same as real ubuntu system mount 
# refer to https://kb.vmware.com/s/article/60262 for details
.host:/    /media/ax/DATAIN    fuse.vmhgfs-fuse    defaults,allow_other,uid=1000     0    0
```

### Application Entry

#### current user

```    
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

touch /home/ax/.local/share/applications/jetbrains-pycharm.desktop
gedit /home/ax/.local/share/applications/jetbrains-pycharm.desktop

[Desktop Entry]
Version=1.0
Type=Application
Name=PyCharm Professional Edition
Icon=/home/ax/opt/pycharm/bin/pycharm.svg
Exec="/home/ax/opt/pycharm/bin/pycharm.sh" %f
Comment=Python IDE for Professional Developers
Categories=Development;IDE;
Terminal=false
StartupWMClass=jetbrains-pycharm
```

### ex****** passport

***For people getting the white screen error, go to Library > History > Show All History, right click on ex******, and Forget About This Site***

## possible configuration

### 配置ssh

```
sudo apt install openssl-server
dpkg -l | grep ssh  # 应该看到 openssh-server
ps -e | grep ssh  # 应该看到 sshd
```

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


### Setup `fail2ban`

```
sudo apt install fail2ban
sudo vi /etc/fail2ban/jail.local 

[sshd]
enabled = true
banaction = iptables-multiport
maxretry = 5
findtime = 900  # ban ip if fail for 5 times in 900 seconds
bantime = 1500  # ban ip for 1500 seconds
port = 130      # ssh port 130. do not need to specify if using port 22

sudo service fail2ban restart
```

### 新建用户

```
sudo adduser <user>  # 按照提示来，一般默认就行
```

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
