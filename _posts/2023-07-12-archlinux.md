---
layout: post
title: my arch linux configuration
date: 2023-07-11 15:09:00
description: an easy installation and set up for people who want to use arch linux for research
tags: linux install
categories: configuration
---
Follow the official install script to install the arch linux. 
However, there are some options or faults encountered when using the script

1. /usr/lib/python3.10/site-packages/archinstall/lib/configuration.py and `if key == 'disk_encryption' and self._config[key]:`
2. [Connect to WiFi](#connect-to-wifi)
3. archinstall
4. filesystem: `ext4`, no extra partition
5. audio server: pulseaudio
6. additional packages: htop neofetch neovim git firefox
7. When selecting the desktop environment, I chose `kde`, then I change to `awesome`. 
*So, what's desktop environment?* check [here](README_linux.md)

{% highlight c++ linenos %}

int
{% endhighlight %}
