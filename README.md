# MachineLearn

## Installation and Getting Started with Pytorch

This is a simple guide to PyTorch, a machine learning library. Pytorch is mainly used for a variety of different applications but is primarely used for computer vision and natural language processing. Pytorch is also a framework that should be explored when looking towards deep learning. For this guide we will learn about how it works and set up a simple model.

In order to first do this you need to be able to download the libraries. For this you have two options you can either do it through Anaconda, or with Pip, for this guide we will be using pip. To do this first we need to have pip installed, you can use the following line:
```
sudo apt install python3-pip
```
After that, head over to the pytorch website and select the prerequesites in order to download the correct version for our device:

https://pytorch.org/get-started/locally/

At this page you can see that there are a series of boxes to check, for my device as well as program it will work on Linux, through the Pip package, and Python as well as run on the CPU. The Pytorch Build which I chose was Stable (2.3.1). Here is my command.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Once you are done with that you are all set with Pytorch, for this program you will also visualize some data so you also need to install matplot which with pip can be done using the following command:
```
sudo apt-get install python3-matplotlib
```

Once you are ready to go you can run the program!