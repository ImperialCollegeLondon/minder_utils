# Download and process the DRI data


## Installation Instructions:

This was tested on python 3.8.11.

Currently tested on MacOS with M1 silicon. If you use M1 silicon, you must use the conda environment and the miniconda environment in step 1.

I have not included tensorflow, pytorch and scikit-learn within the ```setup.py``` because they all have OS unique installation instructions.


1. Firstly, if attempting to install this package on MacOS with M1 silicon then please follow the instructions here: [Installing Tensorflow on MacOS](https://github.com/apple/tensorflow_macos/issues/153) (Please ask Alex if you need help because this is quite unfriendly). Then you may move to step 3. If you are installing on windows, move to step 2.

2. Install tensorflow 2 using the command here: [Tensorflow Installation Guide](https://www.tensorflow.org/install)
    - Tested with tensorflow 2.6.0 on windows and 2.4.0 on MacOS.

3. Install pytorch, using the command here: [Pyorch Installation Guide](https://pytorch.org/get-started/locally/)
    - Tested with pytorch 1.9.1 with cuda 10.2 on windows and 1.9.1 on MacOS.

4. Install scikit-learn using the command here: [Scikit-Learn Installation Guide](https://scikit-learn.org/stable/install.html)
    - Tested with scikit-learn 1.0 on windows and 1.0 on MacOS.

4. Run ```pip install -e git+https://github.com/alexcapstick/minder_utils.git#egg=minder_utils```. There is a notebook, [Install Example.ipynb](https://github.com/alexcapstick/minder_utils/blob/main/Install%20Example.ipynb) with an example of this running in a jupyter notebook. This will save the package in the working directory. Any changes then made to this code will be reflected in your installation.


Please let me know if you run into any issues!

A getting started guide exists here: [Getting Started.ipynb](https://github.com/alexcapstick/minder_utils/blob/main/Getting%20Started.ipynb).


## Troubleshooting:

There are many issues that arise from incompatibilities with Apple's M1 silicon.

- If you are a MacOS M1 user and ran ```conda install jupyterlab``` and it won't work when you run ```jupyter lab```. You also need to run: ```conda install nbclassic==0.2.8```.
- If scikit-learn keeps erroring, uninstall it and then do the following:
    - ``` conda install scikit-learn```
    - ``` conda install scipy```
- After installing the packages on Windows 10, I had to install ```six==1.15.0```, ```typing-extensions==3.7.4``` and ```scipy```, because tensorflow was erroring. 


