#!/bin/bash

source activate python3
pip install --upgrade pip
pip install torch==1.2.0
pip install torchvision==0.4.0
pip install pandas==0.25.3
pip install efficientnet_pytorch
pip install torchsummary
pip install tensorboard
pip install tensorboardX
pip install opencv-python
cd ~
cd auto-deep-learning/src
python3 DeepLearning_GridSearch.py
