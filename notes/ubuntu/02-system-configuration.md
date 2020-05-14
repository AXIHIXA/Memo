# Ubuntu System Configuration

## `.bashrc` 

```
-rw-r--r--  1 ax   ax   3771 Mar 31 13:58 .bashrc
```

```
chmod 644 ~/.bashrc ~/.bash_history
```

## dependencies

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
export EDITOR=/usr/bin/vim
```

## time fix

```
sudo timedatectl set-local-rtc 1 --adjust-system-clock
```
    
## VMWare Shared Folder

```
sudo gedit /etc/fstab

# Use shared folders between VMWare guest and host
# Keep same as real ubuntu system mount 
# refer to https://kb.vmware.com/s/article/60262 for details
.host:/    /media/ax/DATAIN    fuse.vmhgfs-fuse    defaults,allow_other,uid=1000     0    0
```

## Disable `apt-daily-update`

```
Show Applications => Updates => Automatically check for updates: Never

sudo systemctl disable apt-daily.service
sudo systemctl disable apt-daily.timer
sudo systemctl disable apt-daily-upgrade.service
sudo systemctl disable apt-daily-upgrade.timer

sudo gedit /etc/apt/apt.conf.d/20auto-upgrades

APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Download-Upgradeable-Packages "0";
APT::Periodic::AutocleanInterval "0";
APT::Periodic::Unattended-Upgrade "0";
```

## Application Entry

### current user

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

## ex****** passport

For people getting the white screen error, go to Library > History > Show All History, right click on ex******, and Forget About This Site

## 🌱 Some Alienware Stuff

1. `Fn + F6`: disables / enables WIN key
2. When the touchpad fails to respond: reinstall BIOS `Alienware_15_R3_17_R4_1.9.0.EXE`