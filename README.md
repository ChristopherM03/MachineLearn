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

# Using the code
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
The basic premise of the code is to use data which is already in the library for Pytorch and visualize it, then create a small nueral network and train the model, and be able to see not only the accuracy but also the loss for the actual nueral network model.

# For teachers
To learn more you can click on the following [link](https://markgalassi.codeberg.page/small-courses-html/) and click on the Machine Learning chapter (insert chapter number). Additionally this website offers a wide variety of other courses to follow!

# For students
If you would like to understand the code more and follow it step by step, click the following documentation link (insert pdf for code). This is a step by step guide of how to start cloning your repo as well as looking at how the code works. 

# Acknowledgment
This is a project that I worked with the Institue for computing in Research which you can click at this [link](https://computinginresearch.org/). The insitute focuses on allowing students to use advanced computing methods for research and scholarship. To learn more click on the link provided.