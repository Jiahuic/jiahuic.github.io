---
layout: post
title: 'Setting up Arch Linux for Research: A Comprehensive Guide'
date: 2023-07-11 15:09:00
description: an easy installation and set up for people who want to use arch linux for research
tags: linux install
categories: configuration
---
### Introduction
Welcome to this comprehensive guide on setting up Arch Linux for research. This blog post is intended for anyone interested in using Arch Linux for their research activities. We'll walk you through the installation process, highlighting common issues and their solutions. Additionally, we'll guide you through configuring your system for optimal performance. Let's get started!

**NOTE**: it is highly recommended to read the arch wiki.

### Pre-installation Tips and Troubleshooting
Before we dive into the installation process, there are a few things you need to know:

1. **Configuration Updates:** In the installer, you might encounter some issues related to the `self._config[key]` in the `/usr/lib/python3.10/site-packages/archinstall/lib/configuration.py` file. To avoid this, add `if key == 'disk_encryption' and self._config[key]:` to avoid faults before the installation.
2. **Connecting to WiFi:** During the installation, you may need to connect to WiFi. We'll explain how to do this in a later [section](#connect-to-wifi), so for now, just be aware that this is a step you'll need to take.
3. **Running the Installation Script:** To begin the installation process, you'll need to type `archinstall` in the terminal.
4. **Choosing the Filesystem and Partition:** When the script prompts you to choose a filesystem, select ext4. You won't need to set up any extra partitions.
5. **Selecting an Audio Server:** For the audio server, choose pulseaudio.
6. **Installing Additional Packages:** You'll also be prompted to install additional packages. Here, select: `htop`, `neofetch`, `neovim`, `git`, and `firefox`.
7. **Choosing a Desktop Environment:** When it comes to selecting the desktop environment, I recommend choosing `awesome`. If you're unsure what a desktop environment is, check out this [resource](README_linux.md).

That's it for the pre-installation tips and troubleshooting. In the next section, we'll dive into the things you need to do after installing Arch Linux.


<!-- {% highlight c++ linenos %} -->
<!---->
<!-- int -->
<!-- {% endhighlight %} -->
### Things to do after Installation
Once you have succssfully installed Arch Linux, there are a few additional steps to optimize your system for research:
1. **Configure Pacman**: In `/etc/pacman.conf`, uncomment `Color` and `ParallelDownloads`, and add `ILoveCandy`. This will enhance the readability and speed of your package manager.
2. **Install and Configure Zsh**: Install `zsh` and `oh-my-zsh` and make `zsh` your default shell. When you install `oh-my-zsh`, it will prompt you to set it as the default. In your `.zprofile`, add the following code to start X at login:
```
if [ -z "${DISPLAY}" ] && [ "${XDG_VTNR}" -eq 1 ]; then
  exec startx
fi
```
3. **Install Useful Utilities**: install `picom`, `tmux`, `exa`, `zip`, `unzip`, `alsa-utils`, `noto-fonts-cjk`, `exa`, `bat`, `skype`, `discord`, `zoom`, `TexStudio`, `anaconda`, `texlive-core`, `pcmanfm`, `gcc-fortran`, `docker` (`sudo systemctl start docker`), `ruby`, etc.
4. [Check screen tearing](#screen-tearing): Screen tearing can be annoying and disrupt your workflow. Check the later section.
5. [Install font](#install-font):  A good set of fonts can make your system look clean and be easy on the eyes. 
6. **Key setup**: Set up your keys using  `.Xmodmap` with the following configuration:
``` text
clear lock
clear control
add control = Caps_Lock Control_L Control_R
keycode 66 = Control_L Caps_Lock NoSymbol NoSymbol
```
7. **Fix Lenovo T490 Brightness function keys**:  If you are using a Lenovo T490, you might experience issues with brightness function keys. To resolve this, edit `/etc/default/grub` with `GRUB_CMDLINE_LINUX_DEFAULT="quiet pcie_aspm=force acpi_osi="` and then run `$ sudo grub-mkconfig -o /boot/grub/grub.cfg`. Please note that this could potentially cause issues with your system, so proceed with caution.
8. **Audio Control Setup**: Set up audio control by installing `xbindkyes` using the command `sudo pacman -S xbindkeys`. You can find more details about how to use `xbindkeys` [here](https://wiki.archlinux.org/title/Xbindkeys)
9. **Setup xinitrc**: Copy the system `xinitrc` to your home directory using the command ` $ cp /etc/X11/xinit/xinitrc ~/.xinitrc `. 
Add `xbindkeys &` to `.xinitrc` to start `xbindkeys` when you start X.
10. **Enable Touchpad Tapping**: Add `40-libinput.conf` to enable tapping and other toughpad features. Here is a sample configuration:
```
Section "InputClass"
    Identifier "libinput touchpad"
    MatchIsTouchpad "on"
    MatchDevicePath "/dev/input/event*"
    Driver "libinput"
    Option "Tapping" "on"
    Option "NaturalScrolling" "true" # Optional: Enable natural scrolling
EndSection
```
11. **Prevent Accidental Shutdowns**: Configure Display Power Management Signaling (DPMS) to prevent your PC from shutting down accidentally. Check `10-monitor.conf` for the correct settings.
12. **Setup Screen Lock**: Install `i3lock-color` and use this theme `https://github.com/Raymo111/i3lock-color`. There are some dependencies you'll need to check. Use `xautolock` to automatically activate `i3lock`.
13. **Setup File Synchronization**: Install `rslsync` by `yay -S rslsync`. Start Resilio Sync `systemctl start rslsync.service`. Start at boot `systemctl enable rslsync.service`.
You can then access the Resilio Sync interface at `localhost:8888` in your web browser.
You need to add the permission of user's folder `sudo setfacl -R -m "u:rslsync:rwx" /home/your-username` --- I removed it as this current PC doesn't have too much space for download.
14. matlab is at `/usr/local/MATLAB/R2023a`.
15. **Install MeshLab**: from source code, need vcglib (from source code) to replace the empty folder at `src/`
Check to see if the `<cstdint>` header file is included in the `src/external/downloads/libE57Format-2.3.0/include/E57Format.h` and `src/external/downloads/nexus-master/src/corto/src/tunstall.cpp` file. If not, you can add the line `#include <cstdint>` at the top of the file. <br>
`mv /home/{username}/Apps/anaconda3/lib/libstdc++.so.6 /home/{username}/Apps/anaconda3/lib/libstdc++.so.6.bak` <br>
maybe `u3d` need to be installed --- no errors currently
16. **pymol install**: The pymol is installed from its open source version. After the installation, you might have the following error.
``` text
➜  linux git:(main) ✗ pymol
libGL error: MESA-LOADER: failed to open iris: /home/jiahuic/Apps/miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /usr/lib/dri/iris_dri.so) (search paths /usr/lib/dri, suffix _dri)
libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open iris: /home/jiahuic/Apps/miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /usr/lib/dri/iris_dri.so) (search paths /usr/lib/dri, suffix _dri)
libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open swrast: /home/jiahuic/Apps/miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /usr/lib/dri/swrast_dri.so) (search paths /usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
```
To solve the error:
```
mv /home/{username}/Apps/miniconda3/lib/libstdc++.so.6 /home/{username}/Apps/miniconda3/lib/libstdc++.so.6.bak
ln -s /lib/libstdc++.so.6 /home/{username}/Apps/miniconda3/lib/libstdc++.so.6
```

### Connect to WiFi
Enter the `iwd` mode:
``` bash
iwctl # if saying "waiting for iwd to start", do sudo systemctl start iwd
```
In `iwd` mode, you can check the WiFi device by `device list`. 
If you see some devices with power off, you can try
``` bash
adapter phy0 set-property Powered on # some say: device phy0, try both
device wlan0 set-property Powered on
```
Once that done, it is ready to scan and see the WiFi signals 
``` bash
station wlan0 scan
station wlan0 get-networks
station wlan0 connect yourWiFi
station wlan0 show
```


### Conclusion
And there you have it! You've now set up and configured your Arch Linux system, and it's ready for your research endeavors. Remember, the beauty of Linux lies in its flexibility and the control it offers you as a user. Don't hesitate to explore, tweak, and personalize it to suit your needs.

I hope you found this guide helpful. If you run into any issues or have any questions, don't hesitate to leave a comment. Your input could also help other readers who may encounter the same problems.

Thank you for reading and happy researching on Arch Linux!

### Useful References
* [Arch Linux Installation Guide](https://wiki.archlinux.org/title/Installation_guide)
* [Arch Linux General Recommendations](https://wiki.archlinux.org/title/General_recommendations)
* [Zsh Configuration](https://wiki.archlinux.org/title/Zsh)
* [Pacman Tips and Tricks](https://wiki.archlinux.org/title/Pacman/Tips_and_tricks)
* [Xinitrc Configuration](https://wiki.archlinux.org/title/Xinit)
