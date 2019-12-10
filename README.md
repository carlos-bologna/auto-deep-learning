# Auto Deep Learning

Self-service deep learning project helps you to classify images without spend to much work.


# Setup Environment

When you start a EC2 Deep Learning image from AWS, it comes with an old version of Pytorch in a built-in Pytorch environment, so, it's better to use a pre-built Python3 environment and install Pytorch by your own, as following:

```
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
git clone https://github.com/carlos-bologna/auto-deep-learning.git
cd auto-deep-learning

```

# Usefull Commands

```
ssh -i mykey.pem -L 8888:127.0.0.1:8888 ubuntu@xxx.xxx.xxx.xxx
ssh -i mykey.pem -L 6006:127.0.0.1:6006 ubuntu@xxx.xxx.xxx.xxx
sudo mount -a
watch nvidia-smi

# Parameters

You can custom your own set of parameters to apply deep learning grid search in your dataset, to do so, just change ou create a JSON file and put it on parameters directory.
Do not forget to pass the name of the file that you just created, when running the Python file training.

Some parameters allow just a few options, like those:

cuda_devices: [0,1,2,3] or [2,3] or any combination. Use -1 for all available GPUs.

schedulers: set "null" (without quote mark) if you don't want a scheduler, just optimizer.

sample_fracs: fraction of dataset to use. Set to 1.0 to use the entire dataset.

metric: [ACC | KAPPA]

save_best: [metric | loss] if you decide to save the model based on the best score, then, set "loss" (without quote mark) in this parameter.


