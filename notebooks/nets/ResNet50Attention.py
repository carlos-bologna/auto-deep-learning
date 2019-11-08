import torch
import torch.nn as nn
from torchvision.models import resnet50
from nets.blocks import ConvBlock, LinearAttentionBlock, ProjectorBlock

class ResNet50Attention(nn.Module):

    def __init__(self, num_classes, attention=True, pretrained=True, normalize_attn=True):
        super(ResNet50Attention, self).__init__()
        self.attention = attention
        self.num_classes = num_classes
        self.pretrained = pretrained

        resnet50_model = resnet50(pretrained=pretrained)

        layers = [l for l in resnet50_model.children()]

        self.conv1 = layers[0]
        self.bn1 = layers[1]
        self.relu = layers[2]
        self.maxpool = layers[3]

        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.avgpool = layers[8]

        if self.attention:
            self.fc = nn.Linear(in_features=1792, out_features=self.num_classes, bias=True)
        else:
            self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)

        if self.attention:
            self.projector1 = ProjectorBlock(2048, 256)
            self.projector2 = ProjectorBlock(2048, 512)
            self.projector3 = ProjectorBlock(2048, 1024)
            self.attn1 = LinearAttentionBlock(in_features=256, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=1024, normalize_attn=normalize_attn)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.layer4(l3)
        g = self.avgpool(x)
        #x = torch.flatten(g, 1)

        # pay attention
        if self.attention:
            p1 = self.projector1(g)
            c1, g1 = self.attn1(l1, p1)

            p2 = self.projector2(g)
            c2, g2 = self.attn2(l2, p2)

            p3 = self.projector3(g)
            c3, g3 = self.attn3(l3, p3)

            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            x = self.fc(g) # batch_sizexnum_classes

        else:

            x = self.fc(g)

        return x
