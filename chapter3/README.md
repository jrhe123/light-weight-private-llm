# How to install Nvidia on Ubuntu

## 1. Uninstall the default graphic card
```
file: /etc/modprobe.d/blacklist.conf

add two lines into file:
blacklist nouveau
options nouveau modeset = 0

execute: sudo update-initramfs -u

```

## 2. Install driver

```
sudo apt update

apt search nvidia-driver

sudo apt install nvidia-driver-xxx-server

sudo reboot

nvidia-smi
```

# How to install Nvidia on Windows

## 1. Download & install driver

- https://www.nvidia.com/en-us/drivers/

- check status: nvidia-smi


# How to install Nvidia on Windows with WSL & Docker

## 1. Enable WSL on windows

```
dism.exe /online
         /enable-featuer
         /featurename: Microsft-Windows-Subsystem-Linux
         /all /norestart
dism.exe /online
         /enablre-feature
         /featurename:VirtualMachinePlatform
         /all
         /norestart

wsl --install

# we will use WSL version 2
wsl --set-default-version 2
```

## 2. Install docker desktop

- Enable WSL2

## 3. Test service

```
docker run --name my_tensorflow_container \
           --gpus all
           -it
           -p 8888:8888
           tensorflow/tensorflow:latest-gpu-jupyter
```