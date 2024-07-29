# MachineLearn

## Installation and Getting Started with Pytorch
------

This is a simple guide to *PyTorch*, a machine learning library. Pytorch is mainly used for a variety of different applications but is primarely used for computer vision and natural language processing. Pytorch is also a framework that should be explored when looking towards deep learning. For this guide we will learn about how it works and set up a simple model.

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
For our demo lets install one more thing which is NumPy, in the command line run the following::
```
pip install NumPy
```
Once everythin has been downloaded you can now get started with the code.

# What does this code do?
------
First what you will want to do is clone the repo. Now you can use either of the two lines, if you have your SSH key enabaled and set you can use the following command:
```
git clone git@github.com:ChristopherM03/MachineLearn.git
```
If not then you can use the following HTTPS command:
```
git clone https://github.com/ChristopherM03/MachineLearn.git
```

Once you are ready to go and have the repo set up on your machine, you can run the program with the following command line!
```
python3 nnpytorch.py
```
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
These are datasets inside the Pytorch library that you can see will be donwloaded on your device. This is now data that has been downloaded from the torch library which we can use to visualize that data. For the next chunk of code, that will be used to visualize the data:
```Python
figure = plt.figure(figsize=(8, 8))
cols, rows = 5, 5

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```
First we create the figure and then we iterate throughout its columns and rows. We also add the subplot in order to be able to visualize the data. Now after running the code you can visualize a plot of data that are different numbers. This is a simple introduction to using data that Pytorch already has available. Moving on we want our focus now to be on a neural network, in order to do that we need to be able to chunk up the data into batch sizes. We do the following with:
```Python
from torch.utils.data import DataLoader

loaded_train = DataLoader(training_data, batch_size=64, shuffle=True)
loaded_test = DataLoader(test_data, batch_size=64, shuffle=True)
```
