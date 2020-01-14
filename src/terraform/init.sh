#!/bin/bash

# Credentials
AWS_ACCESS_KEY_ID=XXX
AWS_SECRET_ACCESS_KEY=XXX
AWS_DEFAULT_REGION=us-east-1

# Get Instance Tags
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
aws --region $AWS_DEFAULT_REGION \
ec2 describe-spot-instance-requests \
	--filters "Name=tag:Name,Values=gpu-prod" \
	--filters "Name=state,Values=active" \
	--query 'SpotInstanceRequests[0].Tags' > /home/ubuntu/tags.json

# Get Instance Id
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
aws --region $AWS_DEFAULT_REGION \
ec2 describe-spot-instance-requests \
	--filters "Name=tag:Name,Values=gpu-prod" \
	--filters "Name=state,Values=active" \
	--query 'SpotInstanceRequests[0].InstanceId' | tr -d '"' > /home/ubuntu/instance_id.txt

# Write Tags to Instance
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
aws --region $AWS_DEFAULT_REGION \
ec2 create-tags \
	--resources $(< /home/ubuntu/instance_id.txt) \
	--tags file:///home/ubuntu/tags.json

# Attach Volume to Instance
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
aws --region $AWS_DEFAULT_REGION \
ec2 attach-volume \
    --volume-id vol-02b8126c5b436a1db \
    --instance-id $(< /home/ubuntu/instance_id.txt) \
    --device /dev/sdf

# Hold on a second...
sleep 10

# Mount Volume
echo '/dev/xvdf1 /mnt ext4 defaults 0 0' | sudo tee --append /etc/fstab
sudo mount -a

# Clone Repository
cd /home/ubuntu
git clone https://github.com/carlos-bologna/auto-deep-learning.git

# Add Permition to Repository
chown -R ubuntu: auto-deep-learning
cd auto-deep-learning/src

# Initiate training using the tensorflow_36 conda environment
sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate python3; pip install --upgrade pip; pip install torch==1.2.0; pip install torchvision==0.4.0; pip install pandas==0.25.3; pip install efficientnet_pytorch; pip install torchsummary; pip install tensorboard; pip install tensorboardX; pip install opencv-python; python3 DeepLearning_GridSearch.py 2>&1 /tmp/auto-deep-learning.log"
