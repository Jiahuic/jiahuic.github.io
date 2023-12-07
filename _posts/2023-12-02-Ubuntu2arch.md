---
layout: post
title: 'Change Ubuntu to Arch Linux'
date: 2023-11-27 15:09:00
description: I changed my Ubuntu Desktop to Arch Linux
tags: linux install
categories: configuration
---

I have a personal desktop using Ubuntu 22.04.
And I did one more step that I followed the *Ubuntu 22.04 Customization* by LinuxScoop to make it more beautiful.
However, I keep having some problems with the Ubuntu Desktop, such as CapsLock changed back, and some function keys disfunctional.
And the most unacceptable thing is the fan is always running at high speed, which makes me feel like I am in a server room.
Also, the tilling window manager has its benefits.
I decided to change my Ubuntu Desktop to Arch Linux.
Without losing the entertainment function, I will install the Arch Linux with the dual boot with the linux mint.

Here is some steps I did before I start to install Arch Linux.
* backup the data
   - [SSH key](#backup-the-ssh-key)
   - Notes folder
   - Documents folder
   - Garage folder
* partition the disk. I will install the Arch Linux with the dual boot with the linux mint.
   1. Clean the existing partitions
      - Run `lsblk` to check the partitions
      - Start `fdisk`, for example `fdisk /dev/sda`
      - Input `p` to print the partitions
      - Input `d` to delete the partitions
      - Input `w` to write the changes
   2. Create the new partitions
      - Run `fdisk /dev/sda`
      - Input `n` to create a new partition
      - For end sector, input `+512M` for the boot partition
      - Input `w` to write the changes
   3. Formatting the Partitions
      - Run `mkfs.fat -F32 /dev/sda1` to format the boot partition
      - Run `mkfs.ext4 /dev/sda2` to format the root partition
   4. Mounting the Partitions
      - Run `mount /dev/sda2 /mnt` to mount the root partition
      - Run `mkdir /mnt/boot` to create the boot directory
      - Run `mount /dev/sda1 /mnt/boot` to mount the boot partition
   5. During the installation, manually change the disk configuration
      - Select `Manual partitioning`
      - Assign Mont Point `/` to the root partition, `/boot` to the boot partition, and `/home` to the home partition
* install the linux mint
   1. I also think about a dual system with the same `/home`, but it might casue more problems. Clearly, you can expect the configuration files are quit different in both systems.
   2. Convert MBR to GPT
      - Run `gdisk /dev/sda`
      - Input `r` to enter the recovery and transformation menu
      - Input `g` to convert MBR to GPT
      - Input `p` to print the partitions
      - Input `w` to write the changes
   3. Install the linux mint without write the boot loader
      - Run `ubiquity -b` to install the linux mint without write the boot loader
   4. Login to the archlinux
      - Uncomment the `GRUB_DISABLE_OS_PROBER=false` in the `/etc/default/grub`
      - Run `sudo grub-mkconfig -o /boot/grub/grub.cfg` to update the grub


#### Backup the ssh key
##### Before Reinstalling Your System

1. **Locate Your SSH Keys**:
   - By default, your private and public SSH keys are stored in `~/.ssh/`.
   - The private key is typically named `id_rsa`, and the public key is typically named `id_rsa.pub` (for RSA keys; other types like ECDSA or Ed25519 will have different file names).

2. **Backup Your SSH Keys**:
   - Copy these files to a secure backup location. You can use an external drive, a USB stick, or a secure cloud storage service.
   - It's critical to keep your private key (`id_rsa`) secure and confidential.

##### After Reinstalling Your System

1. **Restore Your SSH Keys**:
   - Copy the backed-up SSH key files back into the `~/.ssh/` directory in your home folder on the new system.
   - If the `~/.ssh/` directory doesn't exist, you can create it with `mkdir ~/.ssh`.

2. **Set Correct Permissions**:
   - SSH keys require strict file permissions set for security. Set the correct permissions:
     ```bash
     chmod 700 ~/.ssh
     chmod 600 ~/.ssh/id_rsa
     chmod 644 ~/.ssh/id_rsa.pub
     ```
   - Adjust the filenames if your keys have different names.

3. **Check SSH Agent and Add Keys if Necessary**:
   - If you use an SSH agent, add your SSH key to it:
     ```bash
     ssh-add ~/.ssh/id_rsa
     ```

4. **Test Your SSH Key**:
   - Try connecting to a server where your SSH key was previously authorized to ensure it works correctly.

##### Notes

- **Security**: Always handle your SSH keys securely. The private key should never be shared or exposed to anyone.
- **Key Types**: If you're using a different type of key (like ECDSA or Ed25519), the file names will be different. Make sure to backup the correct key files.
- **New Installation**: Ensure that the SSH client is installed on your new system. It typically is installed by default in Ubuntu.
- **e2fsprogs**: the archlinux always has the lastest version of the e2fsprogs, which is used to format the disk. However, the linux mint 20.04 has the old version of the e2fsprogs. 

#### Reference
* This is a very useful blog about how to update the grub [link](https://averagelinuxuser.com/dual-boot-arch-linux-with-linux/)

