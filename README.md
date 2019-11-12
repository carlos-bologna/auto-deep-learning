# Auto Deep Learning

Self-service deep learning project helps you to classify images without spend to much work.


# Setup Environment

When you start a EC2 Deep Learning image from AWS, it comes with an old version of Pytorch in a built-in Pytorch environment, so, it's better to use a pre-built Python3 environment and install Pytorch by your own, as following:

'''
source activate python3
pip install --upgrade pip
pip install torch==1.2.0
pip install torchvision==0.4.0
pip install pandas==0.25.3
pip install efficientnet_pytorch
pip install torchsummary
pip install tensorboardX
'''