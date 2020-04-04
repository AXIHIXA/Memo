# Ubuntu System Management

## 配置ssh

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


## Setup `fail2ban`

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

## 新建用户

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
