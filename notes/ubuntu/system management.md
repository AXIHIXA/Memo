# Ubuntu System Management

### 🌱 配置`ssh`

```
sudo apt install openssl-server
dpkg -l | grep ssh  # 应该看到 openssh-server
ps -e | grep ssh  # 应该看到 sshd
```

- 如果看到`sshd`那说明`ssh-server`已经启动了。
- 如果没有则可以这样启动：

```
sudo /etc/init.d/ssh stop
sudo /etc/init.d/ssh start
```

- 配置相关：
    - `ssh-server`配置文件位于`/etc/ssh/sshd_config`，在这里可以定义`SSH`的服务端口，默认端口是`22`，你可以自己定义成其他端口号，如`222`；
    - 或把配置文件中的`PermitRootLogin without-password`注释掉，再增加一句`PermitRootLogin yes`，然后重启`SSH`服务。

- 此时已经可以`ssh`登录同时支持`sftp`。

- ssh免密码登录：将需要免密码的机器的`ssh`公钥`id_rsa.pub`拷贝至`${HOME}/.ssh/authorized_ssh`。


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
```
hdfs dfs -ls /
```
    - 或者浏览器中打开`http://localhost:9870`，这个是`hadoop`状态页。
    
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