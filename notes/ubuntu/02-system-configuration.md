# Ubuntu System Configuration






## 🌱 `.bashrc` 

```
-rw-r--r--  1 ax   ax   3771 Mar 31 13:58 .bashrc
```

```
chmod 644 ~/.bashrc ~/.bash_history
```






## 🌱 dependencies

```
# do NOT install tmux mlocate xfce4-terminal filemanager-actions nautilus-actions in WSL! 
sudo apt install tmux mlocate xfce4-terminal filemanager-actions nautilus-actions

sudo apt install build-essential git gcc g++ gdb make cmake clang vim

touch ~/.tmux.conf
gedit ~/.tmux.conf

set -g mouse on
set -g status-interval 60
set -g display-time 3000
set -g history-limit 65535

# default editor
sudo update-alternatives --config editor

# default terminal 
sudo update-alternatives --config x-terminal-emulator
```

Adopted from [this](https://askubuntu.com/questions/76712/setting-nautilus-open-terminal-to-launch-terminator-rather-than-gnome-terminal):
The default File Browser in GNOME (ubuntu 20.04) is nautilus. 
The "Open in Terminal" action in nautilus right click context menu is a plugin by GNOME Terminal so it's hard-coded. 
We may add out own action "Open in xfce4 Terminal" via nautilus-actions. 

1. FileManager-Actions Configuration Tool -> Preferences -> Runtime Preferences -> Nautilus menu layout -> disable "Create a root 'FileManager_Actions' menu"; 
2. Define a new action `Open in xfce4 Terminal`, command: 
    - Path: `/bin/xfce4-terminal`
    - Parameters: `--working-directory=%d/%b`
    - Working directory: `%d`
3. Use command `nautilus -q` to restart nautilus. 





## 🌱 `ssh`

- New `ssh`
```
ssh-keygen
sudo ssh-keygen -A
git config --global user.name "Xi Han"
git config --global user.email "..."
```
- **NEVER DO THIS**: Copy existing `ssh`: your OS username and device name should be the same
```
cp -r /media/ax/DATAIN/Downloads/Coding/.ssh ~
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
eval `ssh-agent -s`
ssh-add
```






## 🌱 time fix

```
sudo timedatectl set-local-rtc 1 --adjust-system-clock
```





## 🌱 Locale Settings

- [Set Monday as the first day of week](https://askubuntu.com/questions/6016/how-to-set-monday-as-the-first-day-of-the-week-in-gnome-2-calendar-applet)
```
sudo gedit /etc/default/profile

LANG="en_US.UTF-8"
LC_TIME="en_GB.UTF-8"
```
Software messages in American, but time in British (weeks starting with Mondays). 






## 🌱 VMWare Shared Folder

```
sudo gedit /etc/fstab

# Use shared folders between VMWare guest and host
# Keep same as real ubuntu system mount 
# refer to https://kb.vmware.com/s/article/60262 for details
vmhgfs-fuse    /media/ax/DATAIN    fuse    defaults,allow_other    0    0
```






## 🌱 Disable `apt-daily-update`

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






## 🌱 Application Entry

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






## 🌱 ex****** passport

For people getting the white screen error, go to Library > History > Show All History, right click on ex******, and Forget About This Site






## 🌱 Some Alienware Stuff

1. `Fn + F6`: disables / enables WIN key
2. When the touchpad fails to respond: reinstall BIOS `Alienware_15_R3_17_R4_1.9.0.EXE`






## 🌱 Some Surface Pro 4 Stuff

1. 提高`Surface Pro 4`  /`Surface Book`屏幕亮度的键盘快捷键：`Fn + Del`
2. 降低`Surface Pro 4` / `Surface Book`屏幕亮度的键盘快捷键：`Fn + Backspace`






## 🌱 Some Windows 10 Stuff

- Register key location to modify display names of applications in Control Panel: 
```
Computer\HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall
```
- Call back Windows Photo Viewer: 
```
@echo off&cd\&color 0a&cls

echo 恢复Win10照片查看器

reg add "HKLM\SOFTWARE\Microsoft\Windows Photo Viewer\Capabilities\FileAssociations" /v ".bmp" /t REG_SZ /d PhotoViewer.FileAssoc.Tiff /f

reg add "HKLM\SOFTWARE\Microsoft\Windows Photo Viewer\Capabilities\FileAssociations" /v ".jpeg" /t REG_SZ /d PhotoViewer.FileAssoc.Tiff /f

reg add "HKLM\SOFTWARE\Microsoft\Windows Photo Viewer\Capabilities\FileAssociations" /v ".jpg" /t REG_SZ /d PhotoViewer.FileAssoc.Tiff /f

reg add "HKLM\SOFTWARE\Microsoft\Windows Photo Viewer\Capabilities\FileAssociations" /v ".png" /t REG_SZ /d PhotoViewer.FileAssoc.Tiff /f

echo 请双击或右击图片，选择“照片查看器”即可

pause
```
