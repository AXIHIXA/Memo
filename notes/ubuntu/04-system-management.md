# Ubuntu System Management



### 🌱 mlocate

How to update database: [HERE](https://askubuntu.com/questions/520963/how-come-the-locate-command-doesnt-find-obvious-files)
```
sudo updatedb
```
Note: **NOT** `sudo mlocate updatedb`!



### 🌱 Commonly-used Commands

#### Terminal

- Ctrl + Shift + F1: GUI
- Ctrl + Shift + F2-FX: TTY terminal
- TTY: teletype (black screen, no GUI)
- PTS: GUI terminal @ `/dev/pts/x`

#### `w` & `tty`

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

#### `df`

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

#### `man`

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

#### Redirection & Pipe

- `ls >> out.txt`: Append stdout to `out.txt`
- `ls > out.txt`: Overwrite `out.txt`with stdout
- `ls N> file`: `N` default to 1 (stdout). 
- File descriptors: 
  - stdin: 0
  - stdout: 1
  - stderr: 2

#### 


### 🌱 配置`ssh`

```
sudo ssh-keygen -A
sudo apt install openssh-server
dpkg -l | grep ssh  # 应该看到 openssh-server
ps -e | grep ssh  # 应该看到 sshd
```

- 如果看到`sshd`那说明`ssh-server`已经启动了。
- 如果没有则可以这样启动：

```
sudo /etc/init.d/ssh stop
sudo /etc/init.d/ssh start

sudo service ssh --full-restart
```

- 配置相关：
    - `ssh-server`配置文件位于`/etc/ssh/sshd_config`，在这里可以定义`SSH`的服务端口，默认端口是`22`，你可以自己定义成其他端口号，如`222`；
    - 或把配置文件中的`PermitRootLogin without-password`注释掉，再增加一句`PermitRootLogin yes`，然后重启`SSH`服务。

- 此时已经可以`ssh`登录同时支持`sftp`。

- ssh免密码登录：将需要免密码的机器的`ssh`公钥`id_rsa.pub`拷贝至`${HOME}/.ssh/authorized_ssh`。

#### `WSL`

```
sudo ssh-keygen -A
sudo apt install openssh-server
sudo vi /etc/ssh/sshd_config

# change following things:
# 1. remove the comment on `port 22` (line 15)
# 2. PasswordAuthentication no => PasswordAuthentication yes (line 58)

# sudo ssh-keygen -A
sudo service ssh --full-restart

# e.g. in Windows Powershell
ssh ax@127.0.0.1 -p ...
```

### 🌱 配置`fail2ban`

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

### 🌱 新建用户

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

#### 切换用户

- `su`以及`su -`，`su -l`后面不加用户，则默认切到 `root`
- `su`是不改变当前变量（only `.bashrc` will be sourced）
- `su -`是`su -l`的简写，“模拟了一次登录（login）”，改变为切换到用户的变量（expericene a login process, usually `.bash_profile` and `.bashrc` will be sourced）
- 也就是说`su`只能获得`root`的执行权限，不能获得环境变量；而`su -`是切换到`root`并获得`root`的环境变量及执行权限

### 🌱 伪分布式`hadoop`

`hadoop`需要`jdk 8`。
正式部署应当为`hadoop`单独创建账号，单机伪分布式配置着玩儿一下就不用啦。
真正的分布式安装可以看`https://zhuanlan.zhihu.com/p/77938727`。
安装流程如下：

- 配置`ssh`以及免密登录（参考本文件开头部分）；
- 配环境变量：
```
cd ~/Downloads
wget https://archive.apache.org/dist/hadoop/core/hadoop-3.2.0/hadoop-3.2.0.tar.gz
tar -zxvf hadoop-3.2.0.tar.gz
mv hadoop-3.2.0 /home/ax/opt/

gedit ~/.bashrc

# hadoop
export HADOOP_HOME=/home/ax/opt/hadoop-3.2.0
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
export HADOOP_MAPRED_HOME=${HADOOP_HOME}
export HADOOP_COMMON_HOME=${HADOOP_HOME}
export HADOOP_HDFS_HOME=${HADOOP_HOME}
export YARN_HOME=${HADOOP_HOME}
export PATH=${PATH}:${HADOOP_HOME}/bin

source ~/.bashrc
hadoop version
```
- 单独配置`${HADOOP_CONF_DIR}/hadoop-env.sh`：
```
gedit ${HADOOP_CONF_DIR}/hadoop-env.sh

# jvm
# in case hadoop does not recognize ${JAVA_HOME} exported in `/etc/profile`
# just let it happy
export JAVA_HOME=/opt/jvm/jdk
```
- 修改4个配置文件的`<configuration></configuration>`域：
    - `core-site.xml`，`10497`是`h`和`a`的`ASCII`码，其实随便谁都行：
    ```
    gedit ${HADOOP_CONF_DIR}/core-site.xml
    
    <configuration>
    <property>
      <name>fs.defaultFS</name>
      <value>hdfs://localhost:10497</value>
    </property>
    </configuration>
    ```
    - `hdfs-site.xml`，第一个是`dfs`的备份数目，单机用`1`就行，后面两个是`namenode`和`datanode`的目录：
    ```
    gedit ${HADOOP_CONF_DIR}/hdfs-site.xml
    
    <configuration>
    <property>
      <name>dfs.replication</name>
      <value>1</value>
    </property>
    <property>
      <name>dfs.datanode.data.dir</name>
      <value>file:///home/ax/var/hadoop/hdfs/datanode</value>
    </property>
    <property>
      <name>dfs.namenode.name.dir</name>
      <value>file:///home/ax/var/hadoop/hdfs/namenode</value>
    </property>
    </configuration>
    ```
    - `mapred-site.xml`：
    ```
    gedit ${HADOOP_CONF_DIR}/mapred-site.xml
    
    <configuration>
      <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
      </property>
    </configuration>
    ```
    - `yarn-site.xml`：
    ```
    gedit ${HADOOP_CONF_DIR}/yarn-site.xml
    
    <configuration>
      <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
      </property>
    </configuration>
    ```
- 格式化`HDFS`：
```
hadoop namenode -format
```
- 启动或关闭：
```
${HADOOP_HOME}/sbin/start-all.sh
${HADOOP_HOME}/sbin/stop-all.sh
```
- 测试`hdfs`，不报错：
    - 或者浏览器中打开`http://localhost:9870`，这个是`hadoop`状态页。
```
hdfs dfs -ls /
```
    
## 🌱 `spark`

```
cd ~/Downloads
wget http://apache.mirrors.hoobly.com/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop3.2.tgz
tar -xzvf spark-3.0.0-preview2-bin-hadoop3.2.tgz  
mv spark-3.0.0-preview2-bin-hadoop3.2 ~/opt/

gedit ~/.bashrc

# spark
export SPARK_HOME=/home/ax/opt/spark-3.0.0-preview2-bin-hadoop3.2
export PATH=${SPARK_HOME}/bin:${PATH}
```

```
cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh
gedit $SPARK_HOME/conf/spark-env.sh

export SPARK_MASTER_IP=master
export SPARK_WORKER_MEMORY=4G
```

```
$SPARK_HOME/sbin/start-all.sh
jps
$SPARK_HOME/bin/run-example SparkPi
看看你有没有得到:
Pi is roughly 3.14716
```
