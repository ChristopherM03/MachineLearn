# MachineLearn

## Installation and Getting Started with Pytorch

This is a simple guide to PyTorch, a machine learning library. Pytorch is mainly used for a variety of different applications but is primarely used for computer vision and natural language processing. Pytorch is also a framework that should be explored when looking towards deep learning. For this guide we will learn about how it works and set up a simple model.

In order to first do this you need to be able to download the libraries. For this you have two options you can either do it through Anaconda, or with Pip, for this guide we will be using pip. To do this first we need to have pip installed, you can use the following line:
```
sudo apt install python3-pip
```
To actually install Pytorch you will have to run a command line that is based off of the device you are using. For my device as well as program it will work on Linux, through the Pip package, and Python as well as run on the CPU. When working with Pytroch I could only work with the one that worked on my CPU so that is when I have in this command line. The Pytorch Build which I chose was Stable (2.3.1). Here is my command.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
If you do not have the same device I am using then head over to the pytorch website and select the prerequesites in order to download the correct version for our device:

https://pytorch.org/get-started/locally/

Once you are done with that you are all set with Pytorch, for this program you will also visualize some data so you also need to install matplot which with pip can be done using the following command:
```
sudo apt-get install python3-matplotlib
```

Once you are ready to go you can run the program with the following command line!
```
python3 nnpytorch.py
```

## What does this do? 

To start off all the different libraries that we are using are being put in. This includes all the different torch libraries, as well as data sets and matplotlib in order to visualize it.
```Python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
```

In the next chunk of code, this is a simple way of loading in already created data from the Pytorch library. When first running the program you will be running these two lines and downloading data.

```Python
training_data = datasets.MNIST(root=".", train=True, download=True, transform=ToTensor())

test_data = datasets.MNIST(root=".", train=False, download=True, transform=ToTensor())
```
These are datasets inside the Pytorch library that you can see will be donwloaded on your device. After that is done with the code, create some new lines and add the following line:
```Python
print(training_data[0])
```

