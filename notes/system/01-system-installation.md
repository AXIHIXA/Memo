# Windows

`Shift + F10` to open cmd in Windows Installer. 
```
diskpart
list disk
select disk <num>
detail disk
```

REMEMBER TO TURN OFF FAST STARTUP FOR WINDOWS in `Control Panel/All COntrol Panel Items/Power Options/System Settings`



# Ubuntu Installation

Taken from [https://www.cnblogs.com/deepllz/p/8892628.html](https://www.cnblogs.com/deepllz/p/8892628.html) as personal note. 

- 首先进入BIOS（dell笔记本开机按F2，其他型号电脑请百度）关闭win10的secure boot，然后在电源选项中禁用快速启动（win10的快速启动在电源选项中，具体路径为：电源选项→选择电源按钮的功能→更改当前不可用设置→去掉启用快速启动前面的√，保存即可）
- 然后再进入boot options（DELL笔记本开机按F12），选择从U盘启动
- 在选择install ubuntu之前，按e进入grub编辑页面，将倒数第二行的 quiet splash 替换成 nomodeset（后面安装成功，并利用软件更新器更新了软件后，启动黑屏问题也是这样解决），按F10保存，即可成功进入安装程序
- 安装过程中的磁盘分区（其他步骤直接点继续就可以了，所以说一下如何进行磁盘分区）在进入到安装类型（install type）这一步骤的时候，建议选择其他（something else）手动进行分区，这样可以自己控制磁盘的分配。手动分区要在空闲分区（free space）分配，点击"+"号即可分配，分配的大小类型如下所示:　 
- The size of the `swap` partition is copied from [this link](https://help.ubuntu.com/community/SwapFaq), `8G` is for `64G` memory. 
        
```
partition    file system    size
efi          --             300M
swap         swap           8G
/boot        ext4           1G
/            ext4           remaining
```
- 分区之后要将boot所对应的分区设置为启动引导器，否则启动时操作系统可能找不到引导项导致无法进入ubuntu
- The grub page is default hidden if only ubuntu is detected. To see this page and edit grub after installation, boot with efi floppy inserted. 

# Update NVIDIA drivers: 
 
- This package manager method is recommended by NVIDIA itself. 
- Use GUI: Show Applications -> Software & Updates -> Additional Drivers -> NVIDIA Cooperation
- [https://ubuntu.com/server/docs/nvidia-drivers-installation](https://ubuntu.com/server/docs/nvidia-drivers-installation)        
```
# 1. Install gcc (required)
# 2. Add ppa (for more nvidia driver versions)
$ sudo apt install vim gcc g++ make cmake build-essential
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
$ sudo ubuntu-drivers list
$ sudo ubuntu-drivers install nvidia:535
```

# CUDA Toolkit 

- For compatibility issues, install cuda toolkit whose driver version is identical to nvidia driver, e.g.:
  - CUDA 12.2, default nvidia driver version is 535. 
  - If these versions mismatch, the original driver is installed, a new bundled driver is installed, nouveau is prohibited again and thus corresponding initrms linux module is recompiled.
  - So this process might yield errors, just bypass by installing CUDA with identical driver version.
- **Post-installation Actions** is needed (environment variables)!
  - See [Here](./03-environment-configuration.md)
- [CUDA Toolkit 12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
- [Prepare Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#prepare-ubuntu)
- [Local Repo Installation for Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
- [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
- [Removing CUDA Toolkit and Driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver)
